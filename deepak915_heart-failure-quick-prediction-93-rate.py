import pandas as pd
import warnings

#Suppressing all warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.head()
df.isna().sum()
small_df=df[['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time']]
x = small_df
y = df['DEATH_EVENT']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.2)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
scaler=StandardScaler()
x_train_sc=scaler.fit_transform(x_train)
x_train=pd.DataFrame(x_train)
x_test_sc=scaler.fit_transform(x_test)
x_test=pd.DataFrame(x_test)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
p3=rfc.predict(x_test)
s3=accuracy_score(y_test,p3)
print("Random Forest Classifier Success Rate :", "{:.2f}%".format(100*s3))
from sklearn.neighbors import KNeighborsClassifier
scorelist=[]
for i in range(1,21):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    p5=knn.predict(x_test)
    s5=accuracy_score(y_test,p5)
    scorelist.append(round(100*s5, 2))
print("K Nearest Neighbors Top 5 Success Rates:")
print(sorted(scorelist,reverse=True)[:5])