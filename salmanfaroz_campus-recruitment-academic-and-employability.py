import matplotlib.pyplot as plt

import warnings

import seaborn as sns

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import svm, datasets

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import xgboost

from plotly.offline import plot, iplot, init_notebook_mode

import plotly.graph_objs as go

from plotly.subplots import make_subplots

import plotly.express as px

from statsmodels.formula.api import ols

init_notebook_mode(connected=True)

warnings.filterwarnings("ignore")



%matplotlib inline
ds=pd.read_csv("../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")



ds.isnull().mean()



ds=ds.fillna(0)



ds.nunique()



placed=ds[ds["status"]=="Placed"]

not_placed=ds[ds["status"]=="Not Placed"]



placed



not_placed.isnull().mean()

fig = px.scatter_3d(ds, x='etest_p', y='mba_p', z='degree_p',

              color='status', width=900, height=600)

fig.update_layout(

    title_text="Analysing the UG-Degree,MBA & Etest Marks with Placement Status")

fig.show()
fig = px.scatter(ds, x="ssc_p", y="hsc_p", color="status")

fig.update_layout(

    title_text="Comparing the 10th and 12th Marks with Placement Status ")

fig.show()
ds = pd.get_dummies(ds, columns=['specialisation'])



fin=ds[ds["specialisation_Mkt&Fin"]==1]

hr=ds[ds["specialisation_Mkt&HR"]==1]



sci_pld=len(fin[fin["degree_t"]=="Sci&Tech"][fin["status"]=="Placed"])

sci_no=len(fin[fin["degree_t"]=="Sci&Tech"][fin["status"]=="Not Placed"])

mgmt_pld=len(fin[fin["degree_t"]=="Comm&Mgmt"][fin["status"]=="Placed"])

mgmt_no=len(fin[fin["degree_t"]=="Comm&Mgmt"][fin["status"]=="Not Placed"])

othr_pld=len(fin[fin["degree_t"]=="Others"][fin["status"]=="Placed"])

othr_no=len(fin[fin["degree_t"]=="Others"][fin["status"]=="Not Placed"])



sci_pld1=len(hr[hr["degree_t"]=="Sci&Tech"][hr["status"]=="Placed"])

sci_no1=len(hr[hr["degree_t"]=="Sci&Tech"][hr["status"]=="Not Placed"])

mgmt_pld1=len(hr[hr["degree_t"]=="Comm&Mgmt"][hr["status"]=="Placed"])

mgmt_no1=len(hr[hr["degree_t"]=="Comm&Mgmt"][hr["status"]=="Not Placed"])

othr_pld1=len(hr[hr["degree_t"]=="Others"][hr["status"]=="Placed"])

othr_no1=len(hr[hr["degree_t"]=="Others"][hr["status"]=="Not Placed"])



ds["degree_t"].value_counts()





val=[sci_pld,sci_no,mgmt_pld,mgmt_no, othr_pld, othr_no]



val1=[sci_pld1,sci_no1,mgmt_pld1,mgmt_no1, othr_pld1, othr_no1]

labels = ["Ug-Sci&Tech - Placed","Ug-Sci&Tech - Not Placed", "Ug-Comm&Mgmt - Placed","Ug-Comm&Mgmt -Not Placed",

         "Ug-Others - Placed","Ug-Others - Not Placed"]



fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])



fig.add_trace(go.Pie(labels=labels, values=[sci_pld,sci_no,mgmt_pld,mgmt_no, othr_pld, othr_no], 

                     name="PG - Mrk&Fin"),

              1, 1)

fig.add_trace(go.Pie(labels=labels, values=[sci_pld1,sci_no1,mgmt_pld1,mgmt_no1, othr_pld1, othr_no1],

                     name="PG - Mrk&HR"),

              1, 2)



fig.update_traces(hole=.4, hoverinfo="label+percent+name")



fig.update_layout(

    title_text="UG and PG wise placement analysis " ,

    annotations=[dict(text='Mrk&Fin', x=0.18, y=0.5, font_size=20, showarrow=False),

                 dict(text='Mrk&HR', x=0.82, y=0.5, font_size=20, showarrow=False)])

fig.show()

print("Mrk&Fin students count : ",sum(val),"\nMrk&HR students count :", sum(val1))
ds = pd.get_dummies(ds, columns=['workex'])



y=ds[ds["workex_Yes"]==1]

n=ds[ds["workex_No"]==1]



labels = ['With Experience who got Job','With Experience who not got Job', "No Experience but Job", "No Experience No Job"]

values = [len(y[y["status"]=="Placed"]),len(y[y["status"]=="Not Placed"]),

         len(n[n["status"]=="Placed"]),len(n[n["status"]=="Not Placed"])]



fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])

fig.update_layout(

    title_text="Work Expereince based Placement Status analysis")

fig.show()
x=ds.loc[:,["ssc_p","hsc_p","degree_p","degree_t","etest_p","specialisation_Mkt&Fin","specialisation_Mkt&HR"]]

ds



x["degree_t"] = x["degree_t"].astype('category')





x["degree_t"] = x["degree_t"].cat.codes





ds["status"] = ds["status"].astype('category')

cleanup_nums = {"status":     {"Placed": 1, "Not Placed": 0}}

ds.replace(cleanup_nums, inplace=True)



 





y=ds["status"]





x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=180)







classifier=xgboost.XGBClassifier()



classifier.fit(x_train,y_train)



pred=classifier.predict(x_test)



classifier.score(x_test,y_test)


def plot_confusion_matrix(cm, names, title="XGBOOST Model's - Confusion matrix", cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(names))

    plt.xticks(tick_marks, names, rotation=45)

    plt.yticks(tick_marks, names)

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    

cm = confusion_matrix(y_test, pred)

np.set_printoptions(precision=2)

print('Confusion matrix, without normalization')

print(cm)

plt.figure()

plot_confusion_matrix(cm, ["Not Placed","Placed"])
st=ds.groupby("status")

st.boxplot(column=['degree_p',"mba_p", "etest_p" ])

print("    0 - Not Placed         1 - Placed ")
model = ols("y ~ x", ds).fit()



print(model.summary()) 
sns.set(style="white")

corr = ds.corr()



mask = np.triu(np.ones_like(corr, dtype=np.bool))



f, ax = plt.subplots(figsize=(11, 9))



cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title("Plotting a diagonal correlation matrix")