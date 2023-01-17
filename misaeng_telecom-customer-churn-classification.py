import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.initializers as init
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()
df.info()
# numeric인데 object(categorical)로 되어있는 것 numeric으로 바꿔줌
df['TotalCharges_new']=pd.to_numeric(df.TotalCharges, errors='coerce_numeric')

# NA가 생겼는지 확인
df.loc[pd.isna(df.TotalCharges_new),'TotalCharges']
# 위의 11개 NA값을 원래 값으로 채워줌
TotalCharges_Missing=[488,753,936,1082,1340,3331,3826,4380,5218,6670,6754]
df.loc[pd.isnull(df.TotalCharges_new),'TotalCharges_new']=TotalCharges_Missing
# 변수 바꿔주고 customerID는 뺌
df.TotalCharges=df.TotalCharges_new
df.drop(['customerID','TotalCharges_new'],axis=1,inplace=True)
# cateogirlcal 변수들의 level 확인
df.dtypes=='object'
categorical_var=[i for i in df.columns if df[i].dtypes=='object']
for z in categorical_var:
    print(df[z].name,':',df[z].unique())
# 위에서 "No internet service"같은 것은 "No"로 바꿔줌
Dual_features= ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
for i in Dual_features:
    df[i]=df[i].apply(lambda x: 'No' if x=='No internet service' else x)
df.MultipleLines=df.MultipleLines.apply(lambda x: 'No' if x=='No phone service' else x)
# 확인
for z in [i for i in df.columns if df[i].dtypes=='object']:
    print(df[z].name,':',df[z].unique())
df.SeniorCitizen= df.SeniorCitizen.apply(lambda x : 'No' if x == 0 else 'Yes')
# 시각화를 위해 바꿔줌

continues_var=[i for i in df.columns if df[i].dtypes !='object']
fig , ax = plt.subplots(1,3,figsize=(15,5))
for i , x in enumerate(continues_var):
    ax[i].hist(df[x][df.Churn=='No'],label='Churn=0',bins=30)
    ax[i].hist(df[x][df.Churn=='Yes'],label='Churn=1',bins=30)
    ax[i].set(xlabel=x,ylabel='count')
    ax[i].legend()
categorical_var_NoChurn=categorical_var[:-1]
#Count Plot all Categorical Variables with Hue Churn
fig , ax = plt.subplots(4,4,figsize=(20,20))
for axi , var in zip(ax.flat,categorical_var_NoChurn):
    sns.countplot(x=df.Churn,hue=df[var],ax=axi)

# 2-class cateogorical 변수 -> sklearn 패키지의 함수로 dummy 변수(0,1)로 바꿔줌
label_encoder = LabelEncoder()
for x in [i for i in df.columns if len(df[i].unique())==2]:
    df[x]= label_encoder.fit_transform(df[x])
    
# 3개이상 class의 cateogorical 변수 -> pandas 이용해서 dummy 변수로 바꿔줌 
df = pd.get_dummies(df, columns= [i for i in df.columns if df[i].dtypes=='object'],drop_first=True)

# 확인
[[x, df[x].unique()] for x in [i for i in df.columns if len(df[i].unique())<10]]
X = df[['TotalCharges', 'InternetService_Fiber optic', 'Contract_One year', 'Contract_Two year',
        'PaperlessBilling', 'TechSupport', 'OnlineSecurity', 'PhoneService', 'InternetService_No', 'tenure']]
 ### SVM으로 classification했을 때 변수 중요도가 높은 10개의 변수만 선택
y = df['Churn']                # target 변수 y

# train, text 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Scale 조정!!!
sc = StandardScaler()
X_train = sc.fit_transform(X_train)                 # fit_transform: fit(μ and σ 계산) + transform
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = sc.fit_transform(X_test)                   # X_train에서 fit한 걸로 transform

X_train.head()

### Tuning
# hidden layer 수: 3
# node 수: (24, 12, 8)
# optimizer: Adam
# epoch 수: 100
model = Sequential()   #Initiate DNN Classifier
X_train.shape
n_hidden1, n_hidden2, n_hidden3 = 24, 12, 8
# Hidden Layer1 (He 초기값)
model.add(Dense(n_hidden1, activation='relu', kernel_initializer=init.he_normal(), input_dim=10))
Dropout(0.5)
# Hidden Layer2 (He 초기값)
model.add(Dense(n_hidden2, activation='relu', kernel_initializer=init.he_normal()))
Dropout(0.5)
# Hidden Layer3 (He 초기값)
model.add(Dense(n_hidden3, activation='relu', kernel_initializer=init.he_normal()))
Dropout(0.5)
# output Layer (Xaiver 초기값)
model.add(Dense(1, activation='sigmoid', kernel_initializer=init.glorot_normal()))
Dropout(0.5)
# compiling
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, batch_size=100, epochs=100, validation_data=(X_test, y_test))
# test data로 예측
y_pred_prob = model.predict(X_test)
y_pred_prob.shape
y_pred = (y_pred_prob > 0.3).astype('int')   # 확률로 예측된 값을 0, 1로 바꿔줌
print(classification_report(y_test,y_pred))
# Confusion Matrix
mat_ann = confusion_matrix(y_test, y_pred)
sns.heatmap(mat_ann.T, square=True, annot=True, fmt='d', cbar=False,
          xticklabels=['No','Yes'],
          yticklabels=['No','Yes'] )
plt.xlabel('true label')
plt.ylabel('predicted label')
svm_classifier= SVC(probability=True)
svm_classifier.fit(X_train,y_train)
svm_prob = svm_classifier.predict_proba(X_test)[:,1]
y_pred_svm[np.where(svm_prob>=0.3)]=1
y_pred_svm[np.where(svm_prob<0.3)]=0
#Classification Report
print(classification_report(y_test,y_pred_svm))
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
logis_prob = logreg.predict_proba(X_test)[:,1]
y_pred_logis = np.zeros_like(y_test)
y_pred_logis[np.where(logis_prob>=0.3)]=1
y_pred_logis[np.where(logis_prob<0.3)]=0
#Classification Report
print(classification_report(y_test,y_pred_logis))