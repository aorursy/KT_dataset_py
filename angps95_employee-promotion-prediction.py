import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.offline as py
py.init_notebook_mode(connected=True)
import cufflinks as cf
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import xgboost as xgb
import scikitplot as skplt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn import preprocessing
import warnings
#plt.style.use(['dark_background'])
%matplotlib inline
warnings.filterwarnings('ignore')
direc = "../input/"
train = pd.read_csv(direc + "train_LZdllcl.csv")
test = pd.read_csv(direc + "test_2umaH9m.csv")
train.head()
train.shape
train.describe()  #5-number summary for numerical columns
for i in train.columns:
    print (i + ": "+str(sum(train[i].isnull()))+" missing values")
train = train.dropna()
train.shape
train["high_prev_rating"] = np.where(train["previous_year_rating"]>=3,1,0)
train["low_prev_rating"] = np.where(train["previous_year_rating"]<3,1,0)
train = train.drop(["previous_year_rating"],axis=1)
train.head()
trace1 = go.Bar(
            x=['Not Promoted','Promoted'],
            y=[sum(train["is_promoted"]==0),sum(train["is_promoted"]==1)],
            marker=dict(color=["red","red"]),
            name="Promotion Rate",
    )

trace2 = go.Bar(
            x=['Males','Females'],
            y=[sum(train["gender"]=='m'),sum(train["gender"]=='f')],
            marker=dict(color=["yellow","yellow"]),
            name='Gender'
    )


trace3 = go.Bar(
            x=['Did Not Win','Won Award'],
            y=[sum(train["awards_won?"]==0),sum(train["awards_won?"]==1)],
            marker=dict(color=["green","green"]),
            name='Award won'
    
    )

trace4 = go.Bar(
            x=['Did not meet KPI','Met KPI'],
            y=[sum(train["KPIs_met >80%"]==0),sum(train["KPIs_met >80%"]==1)],
            marker=dict(color=["blue","blue"]),
            name='Met KPI'
    )

trace5 = go.Bar(
            x=['Other','Sourcing','Referred'],
            y=[sum(train["recruitment_channel"]=='other'),sum(train["recruitment_channel"]=='sourcing'),sum(train['recruitment_channel']=='referred')],
            marker=dict(color=["lime","lime","lime"]),
            name='Recruitment Channels'
    )

trace6 = go.Bar(
            x=list(train["education"].unique()),
            y=[sum(train["education"]==i) for i in list(train["education"].unique())],
            marker=dict(color=["purple","purple","purple"]),
            name="Education"
    )
trace7 = go.Histogram(x=train['age'],name="Distribution of Age")

trace8= go.Histogram(x=train['length_of_service'],name="Length of Service")

trace9=go.Histogram(x=train['avg_training_score'],name="Distribution of average training score")

fig = tools.make_subplots(rows=3, cols=3,
                          subplot_titles=["Promotion rate (Training Set): " + str(round(100*(sum(train["is_promoted"]==1)/train.shape[0]),2)) +"%",
                                         "Male: " + str(round(100*(sum(train["gender"]=="m")/train.shape[0]),2)) + "%, Female: " + str(round(100-100*(sum(train["gender"]=="m")/train.shape[0]),2))+"%",
                                         "Award Winning rate: " + str(round(100*(sum(train["awards_won?"]==1)/train.shape[0]),2)) +"%",
                                         "Percent of KPI>80% Rate: " + str(round(100*(sum(train["KPIs_met >80%"]==1)/train.shape[0]),2)) +"%",
                                         "Recruitment Channels",
                                         "Distribution of Education",
                                          "Distribution of Age",
                                          "Length of Service",
                                          "Average Training Score"
                                         ])
for i in fig['layout']['annotations']:
    i['font'] = dict(size=10,color='black')
fig.append_trace(trace1, 1,1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)
fig.append_trace(trace4, 2, 1)
fig.append_trace(trace5, 2, 2)
fig.append_trace(trace6, 2, 3)
fig.append_trace(trace7, 3, 1)
fig.append_trace(trace8, 3, 2)
fig.append_trace(trace9, 3, 3)
fig['layout'].update(height=900, width=900, title="<b>Distribution of Features<b>")
py.iplot(fig)

labels=list(train["department"].unique())
sizes=[sum(train["department"]==x) for x in labels]

trace = go.Pie(labels=labels, values=sizes,textfont=dict(size=13,color="black"))
layout = go.Layout(
    width=750,
    height=400,
    title = "<b>Proportion of employees in each department<b>",
)
fig=go.Figure([trace],layout=layout)
py.iplot(fig)
ax=sns.distplot(train["no_of_trainings"],kde=False)
plt.title("Number of Trainings")
plt.xlabel("Number")
plt.ylabel("Proportion")
for txt in ax.texts:
    txt.set_visible(False)
sns.factorplot(y="age",x="gender", hue="department",data=train,kind="box",size=4,aspect=8/5)
sns.factorplot(y="length_of_service",x="gender", hue="department",data=train,kind="box",size=4,aspect=8/4)
sns.factorplot(y="age",x="gender", hue="education",data=train,kind="box",size=4,aspect=8/5)
sns.factorplot(y="age",x="gender", hue="is_promoted",data=train,kind="box",size=4,aspect=8/5)
sns.factorplot(y="age",x="department", hue="is_promoted",data=train,kind="box",size=4,aspect=8/2)
sns.factorplot(y="age",x="education", hue="is_promoted",data=train,kind="box",size=4,aspect=8/6)
sns.factorplot(y="age",x="is_promoted", hue="education",data=train,kind="box",size=4,aspect=8/6)
sns.factorplot(y="age",x="is_promoted", hue="department",data=train,kind="box",size=4,aspect=8/2)
def promoted_distribution(variable):
    num = len(list(train[variable].unique()))
    data=[]
    for i in range(num):
        data.append(go.Bar(
            x=['Promoted','Not Promoted'],
            y=[train[train['is_promoted']==1][variable].value_counts()[i],train[train['is_promoted']==0][variable].value_counts()[i]],
            name=str(train[train['is_promoted']==1][variable].value_counts().index[i])))
    layout = go.Layout(
        width=500,
        height=400,
        barmode='stack',
        title = "Promotion rate among " + str(variable)
    )

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='stacked-bar')
    
    
def promoted_stacked_bar(variable):
    x1=list(train[variable].unique())
    trace1 = go.Bar(
        x=x1,
        y=[train[train[variable]==x1[i]]["is_promoted"].value_counts()[0] for i in range(len(x1))],
        name='Not Promoted'
    )
    trace2 = go.Bar(
        x=x1,
        y=[train[train[variable]==x1[i]]["is_promoted"].value_counts()[1] for i in range(len(x1))],
        name='Promoted'
    )
    layout = go.Layout(
        width=500,
        height=400,
        barmode='stack',
        title = "Promotion rate among " + str(variable)
    )
    data=[trace1,trace2]

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)

ls = ['department', 'region', 'education', 'gender', 'recruitment_channel', 'KPIs_met >80%', 'awards_won?']
for i in ls:
    promoted_stacked_bar(i)

for cols in ["department","region","education","recruitment_channel","gender"]:
    train[cols] = train[cols].astype('category')
    train[cols] = train[cols].cat.codes
f,ax = plt.subplots(figsize=(12,12))
sns.heatmap(train.corr(), annot=True, linewidths=0.5, fmt= '.2f',ax=ax)
train["KPI&Award"] = np.where(((train["KPIs_met >80%"]==1) & (train["awards_won?"]==1)),1,0)
promoted_stacked_bar("KPI&Award")
train_features_eng = train
train_features_eng = train_features_eng.drop(['employee_id','recruitment_channel','no_of_trainings','gender','length_of_service','region'],axis=1) #features that are not needed
train_features_eng=pd.get_dummies(train_features_eng, columns=["department","education"], prefix=["Dept", "Eduacation"])
train_features_eng.head()
def train_f1(model):
    return round(f1_score(y_train,model.predict(x_train),average='macro'),2)

def test_f1(model):
    return round(f1_score(y_test,model.predict(x_test),average='macro'),2)

def confusion_matrix_model(model_used):
    cm=confusion_matrix(y_test,model_used.predict(x_test))
    col=["Predicted Promoted","Predicted No Promotion"]
    cm=pd.DataFrame(cm)
    cm.columns=["Predicted Promoted","Predicted No Promotion"]
    cm.index=["Actual Promoted","Actual No Promotion"]
    return cm.T

def confusion_matrix_model_train(model_used):
    cm=confusion_matrix(y_train,model_used.predict(x_train))
    col=["Predicted Promoted","Predicted No Promotion"]
    cm=pd.DataFrame(cm)
    cm.columns=["Predicted Promoted","Predicted No Promotion"]
    cm.index=["Actual Promoted","Actual No Promotion"]
    return cm.T

def importance_of_features(model):
    features = pd.DataFrame()
    features['feature'] = x_train.columns
    features['importance'] = model.feature_importances_
    features.sort_values(by=['importance'], ascending=True, inplace=True)
    features.set_index('feature', inplace=True)
    return features.plot(kind='barh', figsize=(6,6))
x1 = train_features_eng.drop(["is_promoted"],axis=1)
y1 = train_features_eng.loc[:,"is_promoted"]
x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.2,random_state=0,stratify=y1)
k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
x_train.head()
#param_grid = dict(C=(0.0001,0.001,0.005,0.01,0.1,0.5))
#log_reg1 = GridSearchCV(LogisticRegression(penalty="l1"),param_grid=param_grid,scoring="f1_macro")
log_reg1=LogisticRegression(penalty="l1",C=0.5)
log_reg1.fit(x_train,y_train)
#print(log_reg1.best_params_)
print ("In-sample F1 Score: " + str(train_f1(log_reg1)))
print ("Test F1 Score: " + str(test_f1(log_reg1)))
#confusion_matrix_model(log_reg1)
#param_grid = dict(C=(0.0001,0.001,0.005,0.01,0.1,0.5,1))
#log_reg2 = GridSearchCV(LogisticRegression(penalty="l2"),param_grid=param_grid,scoring="f1_macro")
log_reg2=LogisticRegression(penalty="l2",C=1)
log_reg2.fit(x_train,y_train)
#print(log_reg2.best_params_)
print ("In-sample F1 Score: " + str(train_f1(log_reg2)))
print ("Test F1 Score: " + str(test_f1(log_reg2)))
#confusion_matrix_model(log_reg2)
#param_grid = dict(C=(0.001,0.01,0.1,0.5,1,2),gamma=(0.001,0.01,0.1,0.5,1,2))
#svc_rbf = GridSearchCV(SVC(kernel="rbf",random_state=0),param_grid=param_grid,scoring="f1_macro")
svc_rbf = SVC(kernel='rbf', gamma=0.001, C=0.01,random_state=0)
svc_rbf.fit(x_train, y_train)
#print(svc_rbf.best_params_)

print ("In-sample F1 Score: " + str(train_f1(svc_rbf)))
print ("Test F1 Score: " + str(test_f1(svc_rbf)))
#confusion_matrix_model(svc_rbf)
#param_grid = dict(n_neighbors=np.arange(10,70),weights=("uniform","distance"),p=(1,2))
#KNN = GridSearchCV(KNeighborsClassifier(),param_grid=param_grid,scoring="f1_macro")
KNN=KNeighborsClassifier(n_neighbors=30,p=1,weights='distance')
KNN.fit(x_train,y_train)
#print(KNN.best_params_)
print ("In-sample F1 Score: " + str(train_f1(KNN)))
print ("Test F1 Score: " + str(test_f1(KNN)))
#confusion_matrix_model(KNN)
#param_grid = dict(max_depth=np.arange(4,10),min_samples_leaf=np.arange(1,8),min_samples_split=np.arange(2,8),max_leaf_nodes=np.arange(30,100,10))
#Dec_tree = GridSearchCV(DecisionTreeClassifier(),param_grid=param_grid,scoring="f1_macro")
Dec_tree=DecisionTreeClassifier(max_depth= 9, max_leaf_nodes= 60, min_samples_leaf= 7, min_samples_split= 2)
Dec_tree.fit(x_train,y_train)
#print(Dec_tree.best_params_)
print ("In-sample F1 Score: " + str(train_f1(Dec_tree)))
print ("Test F1 Score: " + str(test_f1(Dec_tree)))
#confusion_matrix_model(Dec_tree)
#param_grid = dict(max_depth=np.arange(3,10),min_samples_leaf=np.arange(1,10),min_samples_split=np.arange(2,6),max_leaf_nodes=np.arange(50,120,10))
#param_grid = dict(n_estimators = np.arange(50,500,50))
#ranfor = GridSearchCV(RandomForestClassifier(n_estimators=450,max_depth= 9, max_leaf_nodes=110, min_samples_leaf= 1, min_samples_split= 2,random_state=0),param_grid=param_grid,scoring="f1_macro")
#ranfor = GridSearchCV(RandomForestClassifier(max_depth= 7, max_leaf_nodes=100, min_samples_leaf= 6, min_samples_split= 2,random_state=0),param_grid=param_grid,scoring="accuracy")
ranfor = RandomForestClassifier(n_estimators=450,max_depth= 9, max_leaf_nodes=110, min_samples_leaf= 1, min_samples_split= 2,random_state=0)
ranfor.fit(x_train,y_train)
#print(ranfor.best_params_)
print ("In-sample F1 Score: " + str(train_f1(ranfor)))
print ("Test F1 Score: " + str(test_f1(ranfor)))
#confusion_matrix_model(ranfor)
#param_grid = dict(n_estimators=np.arange(50,500,50),max_depth=np.arange(6,12),learning_rate=(0.0001,0.001,0.01,0.1))
#xgclass = GridSearchCV(xgb.XGBClassifier(random_state=0),param_grid=param_grid,scoring="accuracy")
xgclass = xgb.XGBClassifier(max_depth=9, n_estimators=450, learning_rate=0.01)
xgclass.fit(x_train,y_train)
#print(xgclass.best_params_)
print ("In-sample F1 Score: " + str(train_f1(xgclass)))
print ("Test F1 Score: " + str(test_f1(xgclass)))
confusion_matrix_model(xgclass)
importance_of_features(xgclass)
Classifiers=["Logistic Regression (Lasso)","Logistic Regression (Ridge)","SVC (RBF Kernel)","K-Nearest Neighbours","Decision Tree","Random Forest","XGBoost"]
trainf1 = [train_f1(x) for x in [log_reg1,log_reg2,svc_rbf,KNN,Dec_tree,ranfor,xgclass]]
testf1 = [test_f1(x) for x in [log_reg1,log_reg2,svc_rbf,KNN,Dec_tree,ranfor,xgclass]]
cols=["Classifier","Training F1 Score","Test F1 Score"]
pred_results = pd.DataFrame(columns=cols)
pred_results["Classifier"]=Classifiers
pred_results["Training F1 Score"]=trainf1
pred_results["Test F1 Score"]=testf1
pred_results

