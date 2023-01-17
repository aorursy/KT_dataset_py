import pandas as pd
df=pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')
df.isna().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
l1=df.columns
for i in l1:
    df[i]=le.fit_transform(df[i])


x=df.drop('class',axis=1)
y=df['class']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0,stratify=y)

from sklearn.svm import SVC
svm=SVC(gamma='auto')#just initializing for ease here
from sklearn.model_selection import GridSearchCV
gsc=GridSearchCV(svm,{
    'kernel':['linear','rbf'],
    'C':[1,5,25,100],
},cv=4,return_train_score=False)
gsc.fit(x,y)
gsc.cv_results_
results_overview=pd.DataFrame(gsc.cv_results_)
results_overview
results_tidy=results_overview[['param_C','param_kernel','mean_test_score']]
results_tidy
gsc.best_params_
from sklearn.model_selection import RandomizedSearchCV
rscv=RandomizedSearchCV(svm,{
                            'kernel':['linear','rbf'],
                            'C':[1,5,25,100],
                            },n_iter=3)
rscv.fit(x,y)
rscv.best_params_
