# data analysis libraries:
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype 
# data visualization libraries:
import matplotlib.pyplot as plt
import seaborn as sns
# to ignore warnings:
import warnings
warnings.filterwarnings('ignore')
#timer library
from contextlib import contextmanager
import time
#ML libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
def spilit_and_preprocessing():
    print("Data Preprocessing Has Been Started" "\n")
    data= pd.read_csv('Churn_Modelling.csv')
    df= data.copy()
    df= df.drop(['RowNumber','Surname'], axis=1)
    return df
def feature_engineering(df):
    
    print("Feature Engineering Process Has Been Started" "\n")
    agg_list = {"CreditScore":["mean",'std','min','max'],
                "Tenure":["mean",'min', 'max'],
                "Balance":["median",'std','min','max'],
                "NumOfProducts":["mean",'min', 'max'],
                "EstimatedSalary":["median",'std','min','max'],
                "Age":["mean",'min','max'] }
    
    df_gender = df.groupby("Gender").agg(agg_list)
    df_gender.columns = pd.Index([col[0] + "_" + col[1].upper() + '_Gender' for col in df_gender.columns.tolist()])
    df_gender=df_gender.reset_index()

    df_geography = df.groupby("Geography").agg(agg_list)
    df_geography.columns = pd.Index([col[0] + "_" + col[1].upper() + '_Geography' for col in df_geography.columns.tolist()])
    df_geography=df_geography.reset_index()
 
    df_HasCrCard = df.groupby("HasCrCard").agg(agg_list)
    df_HasCrCard.columns = pd.Index([col[0] + "_" + col[1].upper() + '_HasCrCard' for col in df_HasCrCard.columns.tolist()])
    df_HasCrCard=df_HasCrCard.reset_index()

    df_IsActiveMember = df.groupby("IsActiveMember").agg(agg_list)
    df_IsActiveMember.columns = pd.Index([col[0] + "_" + col[1].upper() + '_IsActiveMember' for col in df_IsActiveMember.columns.tolist()])
    df_IsActiveMember=df_IsActiveMember.reset_index()

    df = df.merge(df_gender, how='left', on= 'Gender')
    df = df.merge(df_geography, how='left', on= 'Geography')
    df = df.merge(df_HasCrCard, how='left', on= 'HasCrCard')
    df = df.merge(df_IsActiveMember, how='left', on= 'IsActiveMember')
    print("Categorical Features Aggregations Process has been finished" "\n")
    df['HasCrCard_2'] = df['HasCrCard'].replace(to_replace = 0, value = -1)
    df['IsActiveMember_2'] = df['IsActiveMember'].replace(to_replace = 0, value = -1)
    df['Age_Balance_Ratio'] = df.Balance / df.Age
    df['Age_Cross_NumOfProducts'] = df.Age * df.NumOfProducts
    df['Score_Age_Ratio'] = df.CreditScore / df.Age
    df['Age_Salary_Ratio'] = df.EstimatedSalary / df.Age
    df['Balance_Salary_Ratio'] = df.Balance / df.EstimatedSalary
    df['Score_Balance_Ratio'] = df.Balance / df.CreditScore
    df['Score_Salary_Ratio'] = df.CreditScore / df.EstimatedSalary
    df['Score_HasCrCard_2'] = df.CreditScore * df.HasCrCard_2
    df['Score_Cross_NumOfProducts'] = df.CreditScore * df.NumOfProducts
    df['Balance_NumOfProducts_Ratio'] = df.CreditScore / df.NumOfProducts
    df['Age_Cross_HasCrCard_2'] = df.Age * df.HasCrCard_2
    df['Balance_Cross_HasCrCard_2'] = df.Balance * df.HasCrCard_2
    df['Age_Cross_IsActiveMember_2'] = df.Age * df.IsActiveMember_2
    df['Score_Cross_IsActiveMember_2'] = df.CreditScore * df.IsActiveMember_2
    df['Balance_Cross_IsActiveMember_2'] = df.Balance * df.IsActiveMember_2
    df['Tenure_Cross_Balance'] = df.Balance * df.Tenure
    df['Tenure_Cross_NumOfProducts'] = df.Tenure * df.NumOfProducts
    df['Score_Cross_Tenure'] = df.Tenure * df.CreditScore
    df['Tenure_Salary_Ratio'] = df.Tenure / df.EstimatedSalary
    
    df= df.drop(['IsActiveMember_2', 'HasCrCard_2'], axis=1)    
    print("Cross Accounts Features Process has been finished" "\n")    
    
    df = pd.get_dummies(df, columns = ["Gender"], prefix = ["Gender"], drop_first= True)
    df = pd.get_dummies(df, columns = ["Geography"], prefix = ["Geography"], drop_first= True)
    df = pd.get_dummies(df, columns = ["HasCrCard"], prefix = ["HasCrCard"], drop_first= True)
    df = pd.get_dummies(df, columns = ["IsActiveMember"], prefix = ["IsActiveMember"], drop_first= True)    
    print("Label Encoding Process has been finished" "\n")    
    return df
def training_model_and_predicting(df):
    print("Model fitting process has been started" "\n")    
    # Main test - train data splitting
    target_1=  df[df['Exited']==1]
    test_1= target_1.sample(400)
    target_0=  df[df['Exited']==0]
    test_0= target_0.sample(1600)
    test = pd.concat([test_1, test_0], axis=0)
    df = df.drop(test.index)
    test_data_for_final=test[['CustomerId', 'Exited']]# this is for random sample success score
    test= test.drop('Exited', axis=1)    
    # train test spilit on train data 
    predictors = df.drop(['Exited', 'CustomerId'], axis=1)
    target = df["Exited"]
    x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.20, random_state = 0)
        
    
    # Tuned model fitting with optimum parameters
    catb = CatBoostClassifier(iterations = 300, learning_rate = 0.04, depth = 5)
    catb_tuned = catb.fit(x_train, y_train)
    y_pred = catb_tuned.predict(x_val)
    acc_catb = round(accuracy_score(y_pred, y_val) * 100, 2)
    print('Train Data Success Score: %' + str(acc_catb))     
    
    #set ids as CustomerId and predict Exited# fitting final model with the best model 
    ids = test['CustomerId']
    predictions = catb_tuned.predict(test.drop('CustomerId', axis=1))
    
    #set the output as a dataframe and convert to csv file named submission.csv
    output = pd.DataFrame({ 'CustomerId' : ids, 'Exited': predictions })
    output.to_csv('submission.csv', index=False)    
    
    #Finally, I test the success of the model with 400 samples randomly selected from the test data.
    test_data_last = test_data_for_final.sample(400)
    testing = test_data_last.merge(output, how= 'left', on = 'CustomerId')
    testing.columns= ['CustomerId','Exited_test','Exited_output']
    testing['toplam'] = testing.Exited_test + testing.Exited_output
    a = testing[testing.toplam!=1].count().iloc[0]
    b= testing.count().iloc[0]    
    print('Random Sample Success Score: %' + str(a/b*100))    
    print("Submission file has been created")
    
    #feature importances table
    feature_imp = pd.Series(catb_tuned.feature_importances_,
                        index=x_train.columns).sort_values(ascending=False).iloc[:20]

    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.xlabel('Feature importances scores')
    plt.ylabel('Features')
    plt.title("Feature importances")
    plt.figure(figsize=(15,20))
    plt.tight_layout()
    plt.savefig('Catboost_importances01.png')
def main():    
    with timer("Preprocessing Time"):
        df = spilit_and_preprocessing()        
    with timer("Feature Engineering Time"):
        df = feature_engineering(df)                
    with timer("Modelling Time"):
        training_model_and_predicting(df)
    
if __name__ == "__main__":
    with timer("Full model run"):
        main()