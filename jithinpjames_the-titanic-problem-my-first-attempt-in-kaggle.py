import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Import function to create training and test set splits
from sklearn.model_selection  import train_test_split

# Import function to automatically create polynomial features! 
from sklearn.preprocessing import PolynomialFeatures

# Import Linear Regression and a regularized regression function
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV

# Finally, import function to make a machine learning pipeline
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
import pickle
%matplotlib inline
df = pd.read_csv('../input/train.csv')
df_unkwn = pd.read_csv('../input/test.csv')
df_unknn_passID = df_unkwn["PassengerId"]
def df_preproccess1(df):
    df['Title']= df['Name'].apply(lambda x : x.split(',')[1].split('.')[0])
    
    def profile(df):
        title = df[0]
        sex = df[1]
        if title in [' Dr',' Rev',' Major',' Col',' Sir',' Don', ' Jonkheer', ' Capt']:
            profile = ' MrU'
        elif title in [' Mlle', ' Ms']:
            profile = ' Miss'
        elif title in [' Lady',' Mme', ' the Countess']:
            profile = ' MrsU'
        elif title in [' Master',' Mrs',' Miss',' Mr']:
            profile = title
        else:
            if sex == 'male':
                profile = ' Mr'
            else:
                profile = ' Mrs'
            
        return profile
    
    df['profile']= df[['Title','Sex']].apply(profile,axis=1)
    df['Count'] = df['SibSp']+ df['Parch']
    df['Cabin_new'] = df['Cabin'].fillna('XXX').apply(lambda x: x.split()[0][0])
    df['Embarked'].fillna(value = df['Embarked'].mode()[0],inplace=True)
    df['Fare'].fillna(value = df['Fare'].mean(),inplace=True)
    return df
def age_model(df):
    #preparing the dataset for age prediction
    age_pred= df[df['Age'].notnull()][['Pclass','Fare','profile','Age']]
    age_profile = pd.get_dummies(age_pred['profile'],drop_first=True)
    age_Pclass = pd.get_dummies(age_pred['Pclass'],drop_first=True)
    age_pred.drop(['profile','Pclass'],axis=1,inplace=True)
    age_pred = pd.concat([age_pred,age_profile,age_Pclass],axis=1)
    
    #Segregating target and predictor variables
    age_pred_X = age_pred.drop(labels='Age',axis=1)
    age_pred_Y = age_pred['Age']
    
    # Alpha (regularization strength) of LASSO regression
    lasso_eps = 0.0001
    lasso_nalpha=1000
    lasso_iter=10000

    # Min and max degree of polynomials features to consider
    degree_min = 2
    degree_max = 3

    # Test/train split
    X_train, X_test, y_train, y_test = train_test_split(age_pred_X, age_pred_Y,test_size=.3)
    # Make a pipeline model with polynomial transformation and LASSO regression with cross-validation, run it for increasing degree of polynomial (complexity of the model)

    for degree in range(degree_min,degree_max+1):
        age_pred_model = make_pipeline(PolynomialFeatures(degree, interaction_only=False), LassoCV(eps=lasso_eps,n_alphas=lasso_nalpha,max_iter=lasso_iter,normalize=True,cv=5))
        age_pred_model.fit(X_train,y_train)
        #test_pred = np.array(model.predict(X_test))
        #RMSE=np.sqrt(np.sum(np.square(test_pred-y_test)))
        #test_score = model.score(X_test,y_test)
    return age_pred_model,age_pred
def age_predict(df,age_pred):
    
    #Creating the new dataframe with missing age values by selecting required variables for age prediction
    age_test = df[df['Age'].isnull()][['Pclass','Fare','profile']]
    age_profile_df = pd.get_dummies(age_test['profile'],drop_first=True)
    age_Pclass_df = pd.get_dummies(age_test['Pclass'],drop_first=True)
    age_test.drop(['profile','Pclass'],axis=1,inplace=True)
    age_test = pd.concat([age_test,age_profile_df,age_Pclass_df],axis=1)
    age_pred,age_test = age_pred.align(age_test, join='outer', axis=1, fill_value=0)
    return age_test
def age_merge(df,age_test,age_pred_model):
    age_fill = age_pred_model.predict(age_test.drop('Age',axis=1))
    Age= pd.DataFrame(age_fill, index=df[df['Age'].isnull()].index,columns=['Age'])
    df = df.merge(Age, how='left',left_index=True, right_index=True)
    df.fillna(value={'Age_x':0,'Age_y':0},inplace=True)
    df['Age'] = df['Age_x'] + df['Age_y'].astype('int64')
    df.drop(labels=['Age_x','Age_y'],inplace=True,axis=1)
    return df   
df = df_preproccess1(df)
age_pred_model,age_pred = age_model(df)
age_test = age_predict(df,age_pred)
df = age_merge(df,age_test,age_pred_model)
df.drop(labels=['Name','PassengerId','Ticket','Cabin','Title'],inplace=True,axis=1)
df_sex = pd.get_dummies(df['Sex'],drop_first=True)
df_embarked = pd.get_dummies(df['Embarked'],drop_first=True)
df_profile = pd.get_dummies(df['profile'],drop_first=True)
df_pclass = pd.get_dummies(df['Pclass'],drop_first=True)

#df_cabin_new = pd.get_dummies(df['Cabin_new'],drop_first=True)
df.drop(['Sex','Embarked','profile','Count','Cabin_new','Pclass'],axis=1,inplace=True)
#df = pd.concat([df,df_sex,df_embarked,df_profile,df_cabin_new],axis=1)
df = pd.concat([df,df_sex,df_embarked,df_profile,df_pclass],axis=1)
X_train = df.drop('Survived',axis=1)
y_train = df['Survived']
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)


log_model = LogisticRegression()
log_model.fit(X_train,y_train)

df_unkwn = df_preproccess1(df_unkwn)
age_unkwn = age_predict(df_unkwn,age_pred)
df_unkwn = age_merge(df_unkwn,age_unkwn,age_pred_model)
df_unkwn.drop(labels=['Name','PassengerId','Ticket','Cabin','Title'],inplace=True,axis=1)
df_unkwn_sex = pd.get_dummies(df_unkwn['Sex'],drop_first=True)
df_unkwn_embarked = pd.get_dummies(df_unkwn['Embarked'],drop_first=True)
df_unkwn_pclass = pd.get_dummies(df_unkwn['Pclass'],drop_first=True)
df_unkwn_profile = pd.get_dummies(df_unkwn['profile'],drop_first=True)
#df_cabin_new = pd.get_dummies(df_unkwn['Cabin_new'],drop_first=True)
df_unkwn.drop(['Sex','Embarked','profile','Cabin_new','Count','Pclass'],axis=1,inplace=True)
df_unkwn = pd.concat([df_unkwn,df_unkwn_sex,df_unkwn_embarked,df_unkwn_profile,df_unkwn_pclass],axis=1)
#df_unkwn = scaler.transform(df_unkwn)
df2,df_unkwn = df.drop('Survived',axis=1).align(df_unkwn, join='outer', axis=1, fill_value=0)
scaler2 = StandardScaler()
scaler2.fit(df_unkwn)
df_unkwn = scaler2.transform(df_unkwn)
#predictions = log_model.predict(df_unkwn)
from sklearn.neural_network import MLPClassifier
mlp_model = MLPClassifier(hidden_layer_sizes=(30,30,30))
mlp_model.fit(X_train,y_train)
predictions = mlp_model.predict(df_unkwn)
#from sklearn.model_selection import GridSearchCV
#from sklearn.svm import SVC
#svc_model = SVC()
#param_grid = {'C':[0.01,0.1,0.5,1,5,10,100],'gamma':[2,1,.5,0.1,.05,0.01,0.001]}
#grid = GridSearchCV(SVC(),param_grid=param_grid,verbose=2)
#grid.fit(X_train,y_train)
#predictions = grid.predict(df_unkwn)
predictions.size
submission = pd.DataFrame({
        "PassengerId": df_unknn_passID,
        "Survived": predictions
    })
submission.to_csv('titanic2.csv', index=False)
submission.info()
