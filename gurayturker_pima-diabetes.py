import pandas as pd 
import numpy as np
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.tools as tls
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix,accuracy_score,mean_squared_error,r2_score,precision_score,f1_score,recall_score,roc_auc_score, roc_curve, classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv("../input/diabetes.csv")
df.head()
df.shape
df.ndim
df.size
df.describe().T
df.count()
df["Outcome"].unique()
diabet = df[(df['Outcome'] != 0)]
healthy = df[(df['Outcome'] == 0)]

data = go.Bar( x = df['Outcome'].value_counts()
              , y = ['Healthy','Diabetic' ], text=df['Outcome'].value_counts(),
                    orientation = 'h',textfont=dict(size=15),
                    textposition = 'auto')


fig = go.Figure(data = data, layout={"title":'Hasta ve Sağlıklı Sayısı'})
py.iplot(fig)

data = go.Pie(labels = ["Healthy","Diabetic"], values = df['Outcome'].value_counts(), 
                   textfont=dict(size=15))
layout = dict(title =  'Sonuç Değişkeni Yüzdelik')
fig = dict(data = data, layout={"title":"Sonuç Değişkeni Yüzdelik"})
py.iplot(fig)
df.cov()
df.corr()
corr=df.corr()

sns.set(font_scale=1.15)
plt.figure(figsize=(14, 10))

sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="black")
plt.title('Değişkenlerin Birbiri ile Korelasyonu');
df.info()
df.isnull().sum()
df[0:30]
df[df["Pregnancies"]==0].count()
df[df["Outcome"]==0].count()
df1=df.copy()
df1
df_nan=df1.iloc[:,1:7].replace(0,np.nan)
df_preg_out_age=df[["Pregnancies","Outcome","Age"]]
df_preg_out_age
df_nan
df2=pd.concat([df_nan,df_preg_out_age],axis=1)
df2.head(20)
df2.isnull().sum()
df2.isnull().sum().sum()
df2.notnull().sum()
df2[df2.isnull().any(axis=1)][0:20]
df2.shape
df2.dropna()
msno.bar(df2)
msno.matrix(df2)
msno.heatmap(df2)
df.corr()
df.cov()
df.describe().T
median_temp=df2[df2["Insulin"].notnull()]
median_temp=median_temp[["Insulin","Outcome"]].groupby(["Outcome"])[["Insulin"]].median().reset_index()
median_temp
mean_temp=df2[df2["Insulin"].notnull()]
mean_temp=mean_temp[["Insulin","Outcome"]].groupby(["Outcome"])[["Insulin"]].mean().reset_index()
mean_temp
def median_for_nan(variable):
    median_temp=df2[df2[variable].notnull()]
    median_temp=median_temp[[variable,"Outcome"]].groupby(["Outcome"])[[variable]].median().reset_index()
    return median_temp
sns.violinplot(y = "Insulin", data = healthy);
sns.distplot(healthy["Insulin"], bins=16, color="purple");
sns.violinplot(y = "Insulin", data = diabet);
sns.distplot(diabet["Insulin"], bins=16, color="purple");
tmp_diabet=diabet["Insulin"]
tmp_healthy=healthy["Insulin"]
hist_data=[tmp_diabet,tmp_healthy]
group_labels = ["Diyabet", "Sağlıklı"]
fig = ff.create_distplot(hist_data, group_labels,  show_hist = True, bin_size =0, curve_type='kde')
fig['layout'].update(title = "Insulin")

py.iplot(fig)
    
median_for_nan("Insulin")
df2.loc[(df2['Outcome'] == 0 ) & (df2['Insulin'].isnull()), 'Insulin'] = 102.5
df2.loc[(df2['Outcome'] == 1 ) & (df2['Insulin'].isnull()), 'Insulin'] = 169.5
sns.violinplot(y="Glucose",data=healthy)
sns.distplot(healthy["Glucose"],bins=16,color="purple")
sns.violinplot(y="Glucose",data=diabet)
sns.distplot(diabet["Glucose"],bins=16,color="purple")
tmp_diabet=diabet["Glucose"]
tmp_healthy=healthy["Glucose"]
hist_data=[tmp_diabet,tmp_healthy]
labels=["Hasta","Sağlıklı"]
fig=ff.create_distplot(hist_data,labels,show_hist = True, bin_size =0, curve_type='kde')
py.iplot(fig)
#her seferinde tekrar yazmayalım
def graph_func(variable):
    tmp_diabet=diabet[variable]
    tmp_healthy=healthy[variable]
    hist_tmp=[tmp_diabet,tmp_healthy]
    layouts=["Hasta","Sağlıklı"]
    fig=ff.create_distplot(hist_tmp,layouts,show_hist=True,bin_size=0,curve_type="kde")
    py.iplot(fig)
median_for_nan("Glucose")
df2.loc[(df2['Outcome'] == 0)  & (df2['Glucose'].isnull()), 'Glucose'] = 107.0
df2.loc[(df2['Outcome'] == 1)  & (df2['Glucose'].isnull()), 'Glucose'] = 140.0
sns.violinplot(y="BloodPressure",data=healthy)
sns.distplot(healthy["BloodPressure"],bins=16,color="purple")
sns.violinplot(y="BloodPressure",data=diabet)
sns.distplot(diabet["BloodPressure"],bins=16,color="purple")
graph_func("BloodPressure")
median_for_nan("BloodPressure")
df2.loc[(df2["Outcome"]==0) & (df2["BloodPressure"].isnull()),"BloodPressure"]=70.0
df2.loc[(df2["Outcome"]==1) & (df2["BloodPressure"].isnull()),"BloodPressure"]=74.5
sns.violinplot(y="SkinThickness",data=healthy)
sns.distplot(healthy["SkinThickness"],bins=16,color="purple")
sns.violinplot(y="SkinThickness",data=diabet)
sns.distplot(diabet["SkinThickness"],bins=16,color="purple")
graph_func("SkinThickness")
median_for_nan("SkinThickness")
df2.loc[(df2["Outcome"]==0)&(df2["SkinThickness"].isnull()),"SkinThickness"]=27.0
df2.loc[(df2["Outcome"]==1)&(df2["SkinThickness"].isnull()),"SkinThickness"]=32.0
sns.violinplot(y="BMI",data=healthy)
sns.distplot(healthy["BMI"],bins=16,color="purple")
sns.violinplot(y="BMI",data=diabet)
sns.distplot(diabet["BMI"],bins=16,color="purple")
graph_func("BMI")
median_for_nan("BMI")
df2.loc[(df2["Outcome"]==0) & (df2["BMI"].isnull()),"BMI"]=30.1
df2.loc[(df2["Outcome"]==1) & (df2["BMI"].isnull()),"BMI"]=34.3
sns.violinplot(y="DiabetesPedigreeFunction",data=healthy)
sns.distplot(healthy["DiabetesPedigreeFunction"],bins=16,color="purple")
sns.violinplot(y="DiabetesPedigreeFunction",data=diabet)
sns.distplot(diabet["DiabetesPedigreeFunction"],bins=16,color="purple")
graph_func("DiabetesPedigreeFunction")
sns.violinplot(y="Age",data=healthy)
sns.distplot(healthy["Age"],bins=16,color="purple")
sns.violinplot(y="Age",data=diabet)
sns.distplot(diabet["Age"],bins=16,color="purple")
graph_func("Age")
sns.violinplot(y="Pregnancies",data=healthy)
sns.distplot(healthy["Pregnancies"],bins=16,color="purple")
sns.violinplot(y="Pregnancies",data=diabet)
sns.distplot(diabet["Pregnancies"],bins=16,color="purple")
graph_func("Pregnancies")
msno.bar(df2)
df2.head(50)
corr=df2.corr()

sns.set(font_scale=1.15)
plt.figure(figsize=(14, 10))

sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="black")
plt.title('Değişkenlerin Birbiri ile Korelasyonu');
sns.lmplot(x="Glucose",y="Insulin",data=df2,col="Outcome",scatter_kws={"color":"green"},line_kws={"color":"red"})
sns.scatterplot(x="Glucose",y="Insulin",data=df2,hue="Outcome")
sns.jointplot(x=df2["Glucose"],y=df2["Insulin"],data=df2,joint_kws={"s":5,"color":"red"},marginal_kws={"color":"red"})
sns.lmplot(x="Glucose",y="BloodPressure",data=df2,col="Outcome",scatter_kws={"color":"green"},line_kws={"color":"red"})
sns.scatterplot(x="Glucose",y="BloodPressure",data=df2,hue="Outcome")
sns.jointplot(x=df2["Glucose"],y=df2["BloodPressure"],joint_kws={"s":5,"color":"red"},marginal_kws={"color":"red"})
sns.lmplot(x="BloodPressure",y="Insulin",data=df2,col="Outcome",scatter_kws={"color":"green"},
          line_kws={"color":"red"})
sns.scatterplot(x = "BloodPressure", y = "Insulin", hue = "Outcome", data = df2);
sns.jointplot(x=df2["BloodPressure"],y=df2["Insulin"],joint_kws={"s":5,"color":"red"},marginal_kws={"color":"red"})
sns.lmplot(x="BloodPressure",y="Age",data=df2,col="Outcome",scatter_kws={"color":"green"},
          line_kws={"color":"red"})
sns.scatterplot(x = "Age", y = "BloodPressure", hue = "Outcome", data = df2);
sns.jointplot(x=df2["Age"],y=df2["BloodPressure"],joint_kws={"s":5,"color":"red"},marginal_kws={"color":"red"})
sns.lmplot(x="BloodPressure",y="SkinThickness",data=df2,col="Outcome",scatter_kws={"color":"green"},
          line_kws={"color":"red"})
sns.scatterplot(x = "BloodPressure", y = "SkinThickness", hue = "Outcome", data = df2);
sns.jointplot(x=df2["BloodPressure"],y=df2["SkinThickness"],joint_kws={"s":5,"color":"red"},marginal_kws={"color":"red"})
sns.lmplot(x="SkinThickness",y="Insulin",data=df2,col="Outcome",scatter_kws={"color":"green"},line_kws={"color":"red"})
sns.scatterplot(x="Insulin",y="SkinThickness",hue="Outcome",data=df2)
sns.jointplot(x=df2["SkinThickness"],y=df2["Insulin"],joint_kws={"s":5,"color":"red"},marginal_kws={"color":"red"})
sns.lmplot(x="Glucose",y="SkinThickness",data=df2,col="Outcome",scatter_kws={"color":"green"},line_kws={"color":"red"})
sns.scatterplot(x="Glucose",y="SkinThickness",hue="Outcome",data=df2)
sns.jointplot(x=df2["Glucose"],y=df2["SkinThickness"],joint_kws={"s":5,"color":"red"},marginal_kws={"color":"red"})
sns.lmplot(x="BMI",y="Insulin",data=df2,col="Outcome",scatter_kws={"color":"green"},line_kws={"color":"red"})
sns.scatterplot(x="BMI",y="Insulin",hue="Outcome",data=df2)
sns.jointplot(x=df2["BMI"],y=df2["Insulin"],joint_kws={"s":5,"color":"red"},marginal_kws={"color":"red"})
sns.lmplot(x="BMI",y="Glucose",data=df2,col="Outcome",scatter_kws={"color":"green"},line_kws={"color":"red"})
sns.scatterplot(x="BMI",y="Glucose",hue="Outcome",data=df2)
sns.jointplot(x=df2["BMI"],y=df2["Glucose"],joint_kws={"s":5,"color":"red"},marginal_kws={"color":"red"})
plt.style.use('ggplot')

f, ax = plt.subplots(figsize=(11, 15))

ax.set_facecolor('#fafafa')
ax.set(xlim=(-.05, 400))
plt.ylabel("Değişkenler")
plt.title("Kutu Grafiği")
ax = sns.boxplot(data = df2, 
  orient = 'h', 
  palette = 'Set2')
df2["Insulin"].mean()
df2["Insulin"].max()
df2_diabet=df2[df2["Outcome"]==1]
df2_healthy=df2[df2["Outcome"]==0]
def aykiri_gozlem_incele_diabet(x):
        Q1=df2_diabet[x].quantile(0.25)
        Q3=df2_diabet[x].quantile(0.75)
        IQR=Q3-Q1
        alt_sinir=Q1-1.5*IQR
        ust_sinir=Q3+1.5*IQR
        return alt_sinir,ust_sinir
def aykiri_gozlem_incele_healthy(x):
        Q1=df2_healthy[x].quantile(0.25)
        Q3=df2_healthy[x].quantile(0.75)
        IQR=Q3-Q1
        alt_sinir=Q1-1.5*IQR
        ust_sinir=Q3+1.5*IQR
        return alt_sinir,ust_sinir



   
liste=["Glucose","BloodPressure","Insulin","DiabetesPedigreeFunction","Age","BMI","SkinThickness","Pregnancies"]
dictionary_diabet=dict()

for i in liste:
    
    dictionary_diabet.setdefault(i,aykiri_gozlem_incele_diabet(i))


dictionary_diabet

dictionary_healthy=dict()
for i in liste:
    dictionary_healthy.setdefault(i,aykiri_gozlem_incele_healthy(i))
dictionary_healthy
altsinir_glucose=45.0
ustsinir_glucose=173.0
df2_healthy_glucose=df2_healthy["Glucose"]
aykiri_glucose_df=(df2_healthy_glucose<altsinir_glucose)
df2_healthy_glucose[aykiri_glucose_df]=altsinir_glucose

aykiri_glucose_df_ust=(df2_healthy_glucose>ustsinir_glucose)
df2_healthy_glucose[aykiri_glucose_df_ust]=ustsinir_glucose
(df2_healthy_glucose<altsinir_glucose)|(df2_healthy_glucose>ustsinir_glucose)
sns.boxplot(x=df2_healthy_glucose)
altsinir_diabet_glucose=47.0
ustsinir_diabet_glucose=239.0
df2_diabet_glucose=df2_diabet["Glucose"]
aykiri_glucose_diabet_df=(df2_diabet_glucose<altsinir_diabet_glucose)
df2_diabet_glucose[aykiri_glucose_diabet_df]=altsinir_diabet_glucose
aykiri_diabet_glucose_df_ust=(df2_diabet_glucose>ustsinir_diabet_glucose)
df2_diabet_glucose[aykiri_diabet_glucose_df_ust]=ustsinir_diabet_glucose
(df2_diabet_glucose<altsinir_diabet_glucose)|(df2_diabet_glucose>ustsinir_diabet_glucose)
sns.boxplot(x=df2_diabet_glucose)
altsinir_bloodpressure=41.75
ustsinir_bloodpressure=99.75
df2_healthy_bloodpressure=df2_healthy["BloodPressure"]
aykiri_bloodpressure_healthy_df=(df2_healthy_bloodpressure<altsinir_bloodpressure)
df2_healthy_bloodpressure[aykiri_bloodpressure_healthy_df]=altsinir_bloodpressure
aykiri_bloodpressure_healthy_df_ust=(df2_healthy_bloodpressure>ustsinir_bloodpressure)
df2_healthy_bloodpressure[aykiri_bloodpressure_healthy_df_ust]=ustsinir_bloodpressure
(df2_healthy_bloodpressure<altsinir_bloodpressure)|(df2_healthy_bloodpressure>ustsinir_bloodpressure)
sns.boxplot(x=df2_healthy_bloodpressure)
altsinir_diabet_bloodpressure=47.0
ustsinir_diabet_bloodpressure=103.0
df2_diabet_bloodpressure=df2_diabet["BloodPressure"]
aykiri_diabet_bloodpressure_df=(df2_diabet_bloodpressure<altsinir_diabet_bloodpressure)
df2_diabet_bloodpressure[aykiri_diabet_bloodpressure_df]=altsinir_diabet_bloodpressure
aykiri_diabet_bloodpressure_ust_df=(df2_diabet_bloodpressure>ustsinir_diabet_bloodpressure)
df2_diabet_bloodpressure[aykiri_diabet_bloodpressure_ust_df]=ustsinir_diabet_bloodpressure
sns.boxplot(x=df2_diabet_bloodpressure)
altsinir_skinthickness=8.5
ustsinir_skinthickness=44.5
df2_healthy_skinthickness=df2_healthy["SkinThickness"]
aykiri_skinthickness_healthy_df=(df2_healthy_skinthickness<altsinir_skinthickness)
df2_healthy_skinthickness[aykiri_skinthickness_healthy_df]=altsinir_skinthickness
aykiri_skinthickness_healthy_df_ust=(df2_healthy_skinthickness>ustsinir_skinthickness)
df2_healthy_skinthickness[aykiri_skinthickness_healthy_df_ust]=ustsinir_skinthickness
(df2_healthy_skinthickness<altsinir_skinthickness)|(df2_healthy_skinthickness>ustsinir_skinthickness)
sns.boxplot(x=df2_healthy_skinthickness)
altsinir_diabet_skinthickness=21.0
ustsinir_diabet_skinthickness=45.0
df2_diabet_skinthickness=df2_diabet["SkinThickness"]
aykiri_diabet_skinthickness_df=(df2_diabet_skinthickness<altsinir_diabet_skinthickness)
df2_diabet_skinthickness[aykiri_diabet_skinthickness_df]=altsinir_diabet_skinthickness
aykiri_diabet_skinthickness_ust_df=(df2_diabet_skinthickness>ustsinir_diabet_skinthickness)
df2_diabet_skinthickness[aykiri_diabet_skinthickness_ust_df]=ustsinir_diabet_skinthickness
sns.boxplot(x=df2_diabet_skinthickness)
altsinir_insulin=80.0
ustsinir_insulin=120.0
df2_healthy_insulin=df2_healthy["Insulin"]
aykiri_insulin_healthy_df=(df2_healthy_insulin<altsinir_insulin)
df2_healthy_insulin[aykiri_insulin_healthy_df]=altsinir_insulin
aykiri_insulin_healthy_df_ust=(df2_healthy_insulin>ustsinir_insulin)
df2_healthy_insulin[aykiri_insulin_healthy_df_ust]=ustsinir_insulin
(df2_healthy_insulin<altsinir_insulin)|(df2_healthy_insulin>ustsinir_insulin)
sns.boxplot(x=df2_healthy_insulin)
altsinir_diabet_insulin=169.5
ustsinir_diabet_insulin=169.5
df2_diabet_insulin=df2_diabet["Insulin"]
aykiri_diabet_insulin_df=(df2_diabet_insulin<altsinir_diabet_insulin)
df2_diabet_insulin[aykiri_diabet_insulin_df]=altsinir_diabet_insulin
aykiri_diabet_insulin_ust_df=(df2_diabet_insulin>ustsinir_diabet_insulin)
df2_diabet_insulin[aykiri_diabet_insulin_ust_df]=ustsinir_diabet_insulin
sns.boxplot(x=df2_diabet_insulin)
altsinir_bmi=11.425
ustsinir_bmi=49.6249
df2_healthy_bmi=df2_healthy["BMI"]
aykiri_bmi_healthy_df=(df2_healthy_bmi<altsinir_bmi)
df2_healthy_bmi[aykiri_bmi_healthy_df]=altsinir_bmi
aykiri_bmi_healthy_df_ust=(df2_healthy_bmi>ustsinir_bmi)
df2_healthy_bmi[aykiri_bmi_healthy_df_ust]=ustsinir_bmi
(df2_healthy_bmi<altsinir_bmi)|(df2_healthy_bmi>ustsinir_bmi)
sns.boxplot(x=df2_healthy_bmi)
altsinir_diabet_bmi=19.087
ustsinir_diabet_bmi=50.5875
df2_diabet_bmi=df2_diabet["BMI"]
aykiri_diabet_bmi_df=(df2_diabet_bmi<altsinir_diabet_bmi)
df2_diabet_bmi[aykiri_diabet_bmi_df]=altsinir_diabet_bmi
aykiri_diabet_bmi_ust_df=(df2_diabet_bmi>ustsinir_diabet_bmi)
df2_diabet_bmi[aykiri_diabet_bmi_ust_df]=ustsinir_diabet_bmi
sns.boxplot(x=df2_diabet_bmi)
altsinir_dpf=-0.26825
ustsinir_dpf=1.05975
df2_healthy_dpf=df2_healthy["DiabetesPedigreeFunction"]
aykiri_dpf_healthy_df=(df2_healthy_dpf<altsinir_dpf)
df2_healthy_dpf[aykiri_dpf_healthy_df]=altsinir_dpf
aykiri_dpf_healthy_df_ust=(df2_healthy_dpf>ustsinir_dpf)
df2_healthy_dpf[aykiri_dpf_healthy_df_ust]=ustsinir_dpf
(df2_healthy_dpf<altsinir_dpf)|(df2_healthy_dpf>ustsinir_dpf)
sns.boxplot(x=df2_healthy_dpf)
altsinir_diabet_dpf=-0.43574
ustsinir_diabet_dpf=1.42625
df2_diabet_dpf=df2_diabet["DiabetesPedigreeFunction"]
aykiri_diabet_dpf_df=(df2_diabet_dpf<altsinir_diabet_dpf)
df2_diabet_dpf[aykiri_diabet_dpf_df]=altsinir_diabet_dpf
aykiri_diabet_dpf_ust_df=(df2_diabet_dpf>ustsinir_diabet_dpf)
df2_diabet_dpf[aykiri_diabet_dpf_ust_df]=ustsinir_diabet_dpf
sns.boxplot(x=df2_diabet_dpf)
altsinir_preg=-5.0
ustsinir_preg=11
df2_healthy_preg=df2_healthy["Pregnancies"]
aykiri_preg_healthy_df=(df2_healthy_preg<altsinir_preg)
df2_healthy_preg[aykiri_preg_healthy_df]=altsinir_preg
aykiri_preg_healthy_df_ust=(df2_healthy_preg>ustsinir_preg)
df2_healthy_preg[aykiri_preg_healthy_df_ust]=ustsinir_preg
(df2_healthy_preg<altsinir_preg)|(df2_healthy_preg>ustsinir_preg)
sns.boxplot(x=df2_healthy_preg)
altsinir_diabet_preg=-7.625
ustsinir_diabet_preg=17.375
df2_diabet_preg=df2_diabet["Pregnancies"]
aykiri_diabet_preg_df=(df2_diabet_preg<altsinir_diabet_preg)
df2_diabet_preg[aykiri_diabet_preg_df]=altsinir_diabet_preg
aykiri_diabet_preg_ust_df=(df2_diabet_preg>ustsinir_diabet_preg)
df2_diabet_preg[aykiri_diabet_preg_ust_df]=ustsinir_diabet_preg
sns.boxplot(x=df2_diabet_preg)
altsinir_age=2.0
ustsinir_age=58
df2_healthy_age=df2_healthy["Age"]
aykiri_age_healthy_df=(df2_healthy_age<altsinir_age)
df2_healthy_age[aykiri_age_healthy_df]=altsinir_age
aykiri_age_healthy_df_ust=(df2_healthy_age>ustsinir_age)
df2_healthy_age[aykiri_age_healthy_df_ust]=ustsinir_age
(df2_healthy_age<altsinir_age)|(df2_healthy_age>ustsinir_age)
sns.boxplot(x=df2_healthy_age)
altsinir_diabet_age=4.0
ustsinir_diabet_age=68.0
df2_diabet_age=df2_diabet["Age"]
aykiri_diabet_age_df=(df2_diabet_age<altsinir_diabet_age)
df2_diabet_age[aykiri_diabet_age_df]=altsinir_diabet_age
aykiri_diabet_age_ust_df=(df2_diabet_age>ustsinir_diabet_age)
df2_diabet_age[aykiri_diabet_age_ust_df]=ustsinir_diabet_age
sns.boxplot(x=df2_diabet_age)
df2_diabet.head(50)
df2_healthy.head(50)
frames=[df2_diabet,df2_healthy]
son_df=pd.concat(frames,ignore_index=True)
son_df
son_df.describe().T
corr=son_df.corr()

sns.set(font_scale=1.15)
plt.figure(figsize=(14, 10))

sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="black")
plt.title('Değişkenlerin Birbiri ile Korelasyonu');
sns.pairplot(son_df, hue = "Outcome", palette="Set2");
son_df.head(50)
son_df["Outcome"].value_counts()
y=son_df["Outcome"]
X=son_df.drop(["Outcome"],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train
knn_model=KNeighborsClassifier().fit(X_train,y_train)
knn_model
y_pred = knn_model.predict(X_test)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
knn = KNeighborsClassifier()
np.arange(1,50)
knn_params = {"n_neighbors": np.arange(1,50)}
knn_cv_model = GridSearchCV(knn, knn_params, cv = 10).fit(X_train, y_train)
knn_cv_model.best_score_
knn_cv_model.best_params_
knn_tuned = KNeighborsClassifier(n_neighbors = 1).fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
knn_tuned.score(X_test, y_test)
accuracy_score(y_test, y_pred)
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)
cart_model = DecisionTreeClassifier().fit(X_train, y_train)
cart_model
y_pred = cart_model.predict(X_test)
accuracy_score(y_test, y_pred)
print(classification_report(y_test,y_pred))
cart = DecisionTreeClassifier()
cart_params = {"max_depth": [1,3,5,8,10],
              "min_samples_split": [2,3,5,10,20,50]}
cart_cv_model = GridSearchCV(cart, cart_params, cv = 10, n_jobs = -1, verbose =2).fit(X_train, y_train)
cart_cv_model.best_params_
cart_tuned = DecisionTreeClassifier(max_depth = 5, min_samples_split = 20).fit(X_train, y_train)
y_pred = cart_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))
y=son_df["Outcome"]
X=son_df.drop(["Outcome"],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.75,random_state=42)
gnb_model=GaussianNB().fit(X_train,y_train)
gnb_model
y_pred=gnb_model.predict(X_test)
y_test
y_pred
cross_val_score(gnb_model,X_train,y_train,cv=10).mean()
accuracy_score(y_test,y_pred)
recall_score(y_test,y_pred)
precision_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))
