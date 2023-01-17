!pip install pycaret
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as plx
from pycaret.classification import *
import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
train.head()
print("Rows and columns in the training set are:-",train.shape[0],",",train.shape[1])
print("Rows and columns in the test set are:-",test.shape[0],",",test.shape[1])
train.info()
fig = plx.pie(names=train["Sex"].value_counts().index,values=train["Sex"].value_counts(),width=500,height=500)

fig.update_layout(
    title={
        'text': "Gender Split",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

fig.show()
fig=plx.bar(train.groupby(["Sex","Survived"])["PassengerId"].count().reset_index().rename(columns={"PassengerId":"Count"}),y="Count",x="Sex",color="Survived",orientation="v")
fig.update(layout_coloraxis_showscale=False)
train["Age_bin"] = pd.cut(train.Age,[0,18,30,40,50,60,100])
train["Age_bin"]=train["Age_bin"].astype("str")
df=train.groupby(["Age_bin","Survived"])["PassengerId"].count().reset_index().rename(columns={"PassengerId":"Count"})
df=df[df["Age_bin"]!="nan"]
fig=plx.bar(df,y="Count",x="Age_bin",color="Survived",orientation="v")
fig.update(layout_coloraxis_showscale=False)
df=train.groupby(["Pclass","Survived"])["PassengerId"].count().reset_index().rename(columns={"PassengerId":"Count"})
df['Pclass']= df['Pclass'].map({1:"1st Class",2:"2nd Class",3:"3rd Class"})
df['Survived']= df['Survived'].map({0:"Not Survived",1:"Survived"})

fig = plx.treemap(df, path=['Survived', 'Pclass'], values='Count')
fig.show()
df=train.groupby(["Embarked","Survived"])["PassengerId"].count().reset_index().rename(columns={"PassengerId":"Count"})
fig = plx.funnel(df, x='Count', y='Embarked', color='Survived')
fig.show()
df=train.groupby(["Pclass","Survived","Embarked"])["PassengerId"].count().reset_index().rename(columns={"PassengerId":"Count"})
df['Pclass']= df['Pclass'].map({1:"1st Class",2:"2nd Class",3:"3rd Class"})
df['Survived']= df['Survived'].map({0:"Not Survived",1:"Survived"})

fig = plx.treemap(df, path=['Survived', 'Embarked','Pclass'], values='Count')
fig.show()
train=train.drop(["PassengerId","Name","Cabin","Ticket"],axis=1)
clf_1 = setup(data = train.dropna(), target = 'Survived', session_id=10001)
best_model = compare_models()
cat_boost = create_model('catboost')
tuned_catboost = tune_model(cat_boost)
boosted_dt = ensemble_model(cat_boost, method = 'Boosting', n_estimators = 100)
interpret_model(cat_boost,plot="summary")
final_cat_boost = finalize_model(cat_boost)
predict_model(final_cat_boost)
#test=test.drop(["PassengerId","Name","Cabin","Ticket"],axis=1)
test["Age_bin"] = pd.cut(test.Age,[0,18,30,40,50,60,100])
test["Age_bin"]=test["Age_bin"].astype("str")
test_predictions = predict_model(final_cat_boost, data=test)
test_predictions.head()
submit = test_predictions[["PassengerId","Label"]].rename(columns={"Label":"Survived"})
submit.to_csv("titanic_submit.csv",index=False)