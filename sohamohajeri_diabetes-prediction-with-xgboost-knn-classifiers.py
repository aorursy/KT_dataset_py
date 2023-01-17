import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.shape
df.describe()
print(df.dtypes.unique())
100*(df.isnull().sum())/(df.shape[0])
df['Outcome'].value_counts()
plt.figure(figsize=(9,9))
plt.pie(x=[500,268], labels=[ 'Diabetic', 'Nondiabetic'], autopct='%1.0f%%',pctdistance=0.6,labeldistance=1.05,textprops={'fontsize':12},colors=['teal','limegreen'])
plt.title('Number of Diabetic and Nondiabetic Patients',loc='center', fontsize=15)
plt.show()
df['Outcome']=df['Outcome'].apply(lambda x: 'Diabetic' if x==1 else 'Nondiabetic')
df.head(2)
sns.pairplot(df,hue='Outcome',palette='viridis')
plt.show()
plt.figure(figsize=(8,6))
corr = df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr, mask=mask, square=True,annot=True, cmap= 'plasma')
plt.title('Correlation Between Features', fontsize=15)
plt.show()
plt.figure(figsize=(15,15))

plt.subplot(4,2,1)
sns.distplot(df['Pregnancies'], color='green')
plt.ylabel('Frequency', fontsize=12)

plt.subplot(4,2,2)
sns.distplot(df['Glucose'], color='blue')
plt.yticks([])

plt.subplot(4,2,3)
sns.distplot(df['BloodPressure'], color='orange')
plt.ylabel('Frequency', fontsize=12)

plt.subplot(4,2,4)
sns.distplot(df['SkinThickness'], color='cyan')
plt.yticks([])

plt.subplot(4,2,5)
sns.distplot(df['Insulin'])
plt.ylabel('Frequency', fontsize=12)

plt.subplot(4,2,6)
sns.distplot(df['BMI'], color='violet')
plt.yticks([])

plt.subplot(4,2,7)
sns.distplot(df['DiabetesPedigreeFunction'], color='forestgreen')
plt.ylabel('Frequency', fontsize=12)

plt.subplot(4,2,8)
sns.distplot(df['Age'], color='royalblue')
plt.yticks([])
plt.show()
ss=StandardScaler()
ss.fit(df.drop(['Outcome'], axis=1))
scaled=ss.transform(df.drop(['Outcome'], axis=1))
scaled_df=pd.DataFrame(data=scaled, columns=df.columns[:-1])
X=scaled_df
y=df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
error_rate=[]

for n in range(1,40):
    knc=KNeighborsClassifier(n_neighbors=n)
    knc.fit(X_train, y_train)
    prediction_knn=knc.predict(X_test)
    error_rate.append(np.mean(prediction_knn!=y_test))
print(error_rate)
plt.figure(figsize=(9,6))
plt.plot(list(range(1,40)), error_rate,color='royalblue', marker='o', linewidth=2, markersize=12, markerfacecolor='deeppink', markeredgecolor='deeppink' )
plt.xlabel('Number of Neighbors', fontsize=12)
plt.ylabel('Error Rate', fontsize=12)
plt.title('Error Rate Versus Number of Neighbors by Elbow Method', fontsize=15)
plt.show()
knc=KNeighborsClassifier(n_neighbors=15)
knc.fit(X_train, y_train)
prediction_knn=knc.predict(X_test)
print(confusion_matrix(y_test,prediction_knn))
print('\n')
print(classification_report(y_test,prediction_knn))
print('Accuracy Score: ',round(accuracy_score(y_test,prediction_knn), ndigits=2))
scaled_df.head()
knc.predict([[0.639947,0.848324,0.149641,0.907270,-0.692891,0.204013,0.468492,1.425995]])
df['Outcome'].iloc[0]
knc.predict([[-0.844885,-1.123396,-0.160546,0.530902,-0.692891,-0.684422,-0.365061,-0.190672]])
df['Outcome'].iloc[3]
X=df.drop(['Outcome'], axis=1)
y=df['Outcome']
X_trian, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
xgbc = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, gamma=0, subsample=0.5,colsample_bytree=1, max_depth=8)
xgbc.fit(X_trian,y_train)
prediction_xgbc=xgbc.predict(X_test)
print(confusion_matrix(y_test,prediction_xgbc))
print('\n')
print(classification_report(y_test,prediction_xgbc))
print('\n')
print('Accuracy Score: ',round(accuracy_score(y_test,prediction_xgbc), ndigits=2))