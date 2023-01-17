# data analysis and wrangling

import numpy as np 

import pandas as pd

from scipy import stats

from scipy.stats import norm, skew

# visualization

import seaborn as sns

import matplotlib. pyplot as plt

import plotly.figure_factory as ff

import plotly.graph_objects as go

# machine learning

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler 

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score,classification_report

from xgboost import XGBClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.feature_selection import f_regression, mutual_info_regression

from xgboost import XGBRegressor

from xgboost import plot_importance
full_data=pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
full_data.head()
all_cols=full_data.columns.values

print(all_cols)
full_data.info()
s = (full_data.dtypes == 'object')

object_cols = list(s[s].index)

print("Categorical variables:")

print(object_cols)
x = (full_data.dtypes == ('int64'))

integer_cols = list(x[x].index)

print("integer variables:")

print(integer_cols)
x = (full_data.dtypes == ('float64'))

float_cols = list(x[x].index)

print("float variables:")

print(float_cols)
print("numerical variables:")

numerical_cols=float_cols+integer_cols

print(numerical_cols)
missing_prop_column=full_data.isnull().mean()

missing_prop_column
#encoding the status column for ease of anlysis

labelencoder = LabelEncoder()

full_data['placed']=labelencoder.fit_transform(full_data['status'])
full_data.describe()
def displot_violinboxplot(col):

 

 col=full_data[col]

 hist_data = [col]

 group_labels = [' Distribution of the variable by displot']

 colors=['red']

 fig1 = ff.create_distplot(hist_data, group_labels,colors=colors,bin_size=[1]) #custom bin_size

 fig1.update_layout(

    autosize=False,

    width=800,

    height=400,)

 

 fig1.show()

 fig2 = go.Figure(data=go.Violin(y=col, box_visible=True, line_color='black',

                               meanline_visible=True, fillcolor='lightseagreen', opacity=0.6,

                               x0='violinboxplot'))

 fig2.update_layout(

    autosize=False,

    width=800,

    height=400,)



 fig2.update_layout(yaxis_zeroline=False,title="Distribution of the column")

 fig2.show()
#visualization of the feature ssc_p

displot_violinboxplot('ssc_p')
#visualization of the feature hsc_p

displot_violinboxplot('hsc_p')
#displaying the distribution of degree_p

displot_violinboxplot('degree_p')
#displaying the distribution of etest_p

displot_violinboxplot('etest_p')
#displaying the distribution of mba_p

displot_violinboxplot('mba_p')
#to get the QQ-plot we have to remove the missing values otherwise there will raise error

#so we are creating no_missing_salary data frame only for graphical purpose

no_missing_salary_df=full_data.dropna()
plt.figure(figsize=(10,5))

sns.distplot(full_data['salary'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(no_missing_salary_df['salary'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('salary distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(no_missing_salary_df['salary'], plot=plt)

plt.show()
salary=full_data['salary']

fig = go.Figure(go.Box(x=salary,name="Salary")) # to get Horizonal plot change axis :  x=germany_score

fig.update_layout(title="Salary Distribution")

fig.show()
#plotting the distribution curves together

ssc_p=full_data['ssc_p']

hsc_p=full_data['hsc_p']

degree_p=full_data['degree_p']

etest_p=full_data['etest_p']

mba_p=full_data['mba_p']		



hist_data = [ssc_p,hsc_p,degree_p,etest_p,mba_p] 

group_labels = ["ssc_p Distribution ","hsc_p Distribution ","degree_p Distribution","etest_p Distribution ","mba_p Distribution "]

colors=['blue',"green","orange","red","black"]

fig = ff.create_distplot(hist_data, group_labels,show_hist=False,colors=colors,bin_size=[1,1,1,1,1])

fig.show()
#plotting the box_plots of different marks distributions together

import plotly.graph_objects as go

ssc_p=full_data['ssc_p']

hsc_p=full_data['hsc_p']

degree_p=full_data['degree_p']

etest_p=full_data['etest_p']

mba_p=full_data['mba_p']		

fig = go.Figure()

fig.add_trace(go.Box(y=ssc_p,

                     marker_color="blue",

                     name="ssc_p"))

fig.add_trace(go.Box(y=hsc_p,

                     marker_color="green",

                     name="hsc_p"))

fig.add_trace(go.Box(y=degree_p,

                     marker_color="orange",

                     name="degree_p"))

fig.add_trace(go.Box(y=etest_p,

                     marker_color="red",

                     name="etest_p"))

fig.add_trace(go.Box(y=mba_p,

                     marker_color="black",

                     name="mba_p"))

fig.update_layout(title="Distribution of different numerical variables")

fig.show()
full_data.describe(include=['O'])
#we are creating a function to display bar chart and pie chart of the differnt categories of a categorical variable.

def pie_bar(col):

  category=full_data[col].value_counts().to_frame().reset_index().rename(columns={'index':col,col:'count'})

  fig1 = go.Figure(go.Bar(

    x=category[col],y=category['count'],

  ))

  fig1.update_layout(title_text=' Bar chart of different categories of the variable',xaxis_title="category",yaxis_title="count")







  fig2= go.Figure([go.Pie(labels=category[col], values=category['count'])])



  fig2.update_traces(textposition='inside', textinfo='percent+label')



  fig2.update_layout(title="Pie chart of different categories of the variable",title_x=0.5)

  print(category)

  fig1.update_layout(

    autosize=False,

    width=800,

    height=400,)

  fig2.update_layout(

    autosize=False,

    width=800,

    height=400,)

 

 

  fig1.show()

  fig2.show()

#gender

pie_bar('gender')
#ssc_b

pie_bar('ssc_b')
pie_bar('hsc_b')
pie_bar('hsc_s')
pie_bar('degree_t')
pie_bar('workex')
pie_bar('specialisation')
pie_bar('status')
#function to check the relationship between different categorical variables and placement

def placement_ratio(col):

  

  result=full_data[[col, 'placed']].groupby([col], as_index=False).mean().sort_values(by='placed', ascending=False)

  return result

placement_ratio('gender')
placement_ratio('ssc_b')
placement_ratio('hsc_b')
placement_ratio('hsc_s')
placement_ratio('degree_t')
placement_ratio('workex')
placement_ratio('specialisation')
fig, axes = plt.subplots(4, 2,  figsize=(15, 10))



full_data['frequency'] = 0 # a dummy column to refer to

for col, ax in zip(['gender', 'ssc_b', 'hsc_b', 'hsc_s','degree_t','workex','specialisation'], axes.flatten()):

    counts = full_data.groupby([col, 'placed']).count()

    freq_per_group = counts.div(counts.groupby(col).transform('sum')).reset_index()

    sns.barplot(x=col, y='frequency', hue='placed', data=freq_per_group, ax=ax)

fig.delaxes(axes[3, 1])    
#function to create histograms categorized by placement 

def hist_cont_place(x):

    %matplotlib inline

    g = sns.FacetGrid(full_data, col='placed')

    g.map(plt.hist, x, bins=20)
#function to create box_plots categorized by placement

def boxplot_cont_place(x):

    got_placement=full_data[full_data['placed']==1][x]

    noplacement=full_data[full_data['placed']==0][x]

    fig = go.Figure()

    fig.add_trace(go.Box(y=got_placement,

                     marker_color="blue",

                     name="placed"))

    fig.add_trace(go.Box(y=noplacement,

                     marker_color="red",

                     name="not placed"))

    fig.update_layout(

    autosize=False,

    width=800,

    height=400,)

    fig.update_layout(title="Distribution of percentage according placement")

    fig.show()
hist_cont_place('ssc_p')
boxplot_cont_place('ssc_p')
hist_cont_place('hsc_p')
boxplot_cont_place('hsc_p')
hist_cont_place('degree_p')
boxplot_cont_place('degree_p')
hist_cont_place('etest_p')
boxplot_cont_place('etest_p')
hist_cont_place('mba_p')
boxplot_cont_place('mba_p')
#function to plot category wise salary histograms of different categorical features.

def catcol_vs_salary(col):

  g= sns.FacetGrid(full_data, col =col, size = 3, aspect = 2)

  g.map(plt.hist, 'salary', color = 'r'), plt.show()

  plt.show()  
#function to plot category wise salary swarmplots of different categorical features.

def swramplot_cat_salary(col):

  sns.set(style="whitegrid")

  ax = sns.swarmplot(x=col, y="salary", data=full_data)
catcol_vs_salary('gender')
sns.kdeplot(full_data.salary[ full_data.gender=="M"])

sns.kdeplot(full_data.salary[ full_data.gender=="F"])

plt.legend(["Male", "Female"])

plt.xlabel("Salary (100k)")

plt.show()
swramplot_cat_salary('gender')
catcol_vs_salary('ssc_b')
sns.kdeplot(full_data.salary[ full_data.ssc_b=="Others"])

sns.kdeplot(full_data.salary[ full_data.ssc_b=="Central"])

plt.legend(["Others", "Central"])

plt.xlabel("Salary (100k)")

plt.show()
swramplot_cat_salary('ssc_b')
catcol_vs_salary('hsc_b')
sns.kdeplot(full_data.salary[ full_data.hsc_b=="Others"])

sns.kdeplot(full_data.salary[ full_data.hsc_b=="Central"])

plt.legend(["Others", "Central"])

plt.xlabel("Salary (100k)")

plt.show()
swramplot_cat_salary('hsc_b')
catcol_vs_salary('hsc_s')
sns.kdeplot(full_data.salary[ full_data.hsc_s=="Commerce"])

sns.kdeplot(full_data.salary[ full_data.hsc_s=="Science"])

sns.kdeplot(full_data.salary[ full_data.hsc_s=="Arts"])

plt.legend(["Commerce", "Science","Arts"])

plt.xlabel("Salary (100k)")

plt.show()
swramplot_cat_salary('hsc_s')
catcol_vs_salary('degree_t')
sns.kdeplot(full_data.salary[ full_data.degree_t=="Sci&Tech"])

sns.kdeplot(full_data.salary[ full_data.degree_t=="Comm&Mgmt"])

sns.kdeplot(full_data.salary[ full_data.degree_t=="Others"])

plt.legend(["Sci&Tech", "Comm&Mgmt","Others"])

plt.xlabel("Salary (100k)")

plt.show()
swramplot_cat_salary('degree_t')
catcol_vs_salary('workex')
sns.kdeplot(full_data.salary[ full_data.workex=="No"])

sns.kdeplot(full_data.salary[ full_data.workex=="Yes"])

plt.legend(["No", "Yes"])

plt.xlabel("Salary (100k)")

plt.show()
swramplot_cat_salary('workex')
catcol_vs_salary('specialisation')
sns.kdeplot(full_data.salary[ full_data.specialisation=="Mkt&HR"])

sns.kdeplot(full_data.salary[ full_data.specialisation=="Mkt&Fin"])

plt.legend(["Mkt&HR", "Mkt&Fin"])

plt.xlabel("Salary (100k)")

plt.show()
swramplot_cat_salary('specialisation')
# creating a function plot the hiscontour of continuous variables vs salary

def histcontour(x):

    x = full_data[x]

    y = full_data['salary']



    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Histogram2dContour(

        x = x,

        y = y,

        colorscale = 'gray',

        reversescale = True,

        xaxis = 'x',

        yaxis = 'y'

        ))

    fig.add_trace(go.Scatter(

        x = x,

        y = y,

        xaxis = 'x',

        yaxis = 'y',

        mode = 'markers',

        marker = dict(

            color = "red", #'rgba(0,0,0,0.3)',

            size = 3

        )

    ))

    fig.add_trace(go.Histogram(

        y = y,

        xaxis = 'x2',

        marker = dict(

            color = "blue", #'rgba(0,0,0,1)'

        )

    ))

    fig.add_trace(go.Histogram(

        x = x,

        yaxis = 'y2',

        marker = dict(

            color = "blue",# 'rgba(0,0,0,1)'

        )

    ))



    fig.update_layout(

    autosize = False,

    xaxis = dict(

        zeroline = False,

       domain = [0,0.85],

        showgrid = False

    ),

    yaxis = dict(

        zeroline = False,

       domain = [0,0.85],

        showgrid = False

    ),

    xaxis2 = dict(

        zeroline = False,

       domain = [0.85,1],

        showgrid = False

    ),

    yaxis2 = dict(

        zeroline = False,

        domain = [0.85,1],

        showgrid = False

    ),

    height = 600,

    width = 600,

    bargap = 0,

    hovermode = 'closest',

    showlegend = False,

    title_text="Density Contour of the variable with salary",title_x=0.5

    )

    fig.show()
# creating a function to plot the lineplot of continuous variables vs salary

def lineplot_numeric_vs_salary(col):

    sns.lineplot(col, "salary", data=full_data)

    plt.figure(figsize=(5,4))

    plt.show()
histcontour('ssc_p')
lineplot_numeric_vs_salary('ssc_p')
histcontour('hsc_p')
lineplot_numeric_vs_salary('hsc_p')
histcontour('degree_p')
lineplot_numeric_vs_salary('degree_p')
histcontour('mba_p')
lineplot_numeric_vs_salary('mba_p')
histcontour('etest_p')
lineplot_numeric_vs_salary('etest_p')
full_data=full_data.drop(['status','frequency','sl_no'], axis = 1)
numeric_full_data= full_data.select_dtypes(['number'])

numeric_full_data= numeric_full_data.drop(['placed'],axis=1)

colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(numeric_full_data.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
#converting the categorical variables into numerical variables 

full_data['gender'] = full_data['gender'].map( {'F': 0, 'M': 1} ).astype(int)

full_data['ssc_b'] = full_data['ssc_b'].map( {'Others': 0, 'Central': 1} ).astype(int)

full_data['hsc_b'] = full_data['hsc_b'].map( {'Others': 0, 'Central': 1} ).astype(int)

full_data['hsc_s'] = full_data['hsc_s'].map( {'Commerce': 0, 'Science': 1,'Arts':2} ).astype(int)

full_data['degree_t'] = full_data['degree_t'].map( {'Sci&Tech': 0, 'Comm&Mgmt': 1,'Others':2} ).astype(int)

full_data['workex'] = full_data['workex'].map( {'No': 0, 'Yes': 1} ).astype(int)

full_data['specialisation'] = full_data['specialisation'].map( {'Mkt&HR': 0, 'Mkt&Fin': 1} ).astype(int)
full_data.head()
full_data = full_data.drop(full_data[(full_data['salary']>390000)].index)

no_missing_salary_df =no_missing_salary_df.drop(no_missing_salary_df[(no_missing_salary_df['salary']>390000)].index)

full_data = full_data.drop(full_data[(full_data['hsc_p']<45.83)].index)

full_data = full_data.drop(full_data[(full_data['hsc_p']>87.6)].index)

full_data = full_data.drop(full_data[(full_data['degree_p']>84)].index)

full_data = full_data.drop(full_data[(full_data['mba_p']>75.71)].index)

skewValue = numeric_full_data.skew(axis=0)

print(skewValue)
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

full_data["salary"] = np.log1p(full_data["salary"])
plt.figure(figsize=(10,5))

sns.distplot(full_data['salary'] , fit=norm);

no_missing_salary_df["salary"] = np.log1p(no_missing_salary_df["salary"])



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(no_missing_salary_df['salary'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Salary')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(no_missing_salary_df['salary'], plot=plt)

plt.show()
salary_full_data=full_data[full_data['placed']==1]
salary_full_data.head()
X = full_data.iloc[:,0:12]  #independent columns

y = full_data.iloc[:,-1]    #target column i.e placed

#apply SelectKBest class to extract top 10 features

bestfeatures = SelectKBest(score_func=chi2, k=12)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['features','Score']  #naming the dataframe columns

print(featureScores.nlargest(12,'Score'))  #print best features
X = full_data[[ 'ssc_p', 'hsc_p',  'degree_p',  'workex','etest_p', 'specialisation']]

y = full_data['placed']
scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3,random_state=1)
# Support Vector Machines

svc = SVC(probability=True)

svc.fit(X_train, y_train)

Y_pred = svc.predict(X_test)

acc_svc=100*accuracy_score(y_test, Y_pred)

acc_svc
print(classification_report(y_test, Y_pred))
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

Y_pred = knn.predict(X_test)

acc_knn=100*accuracy_score(y_test, Y_pred)

acc_knn
print(classification_report(y_test, Y_pred))
gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian =100*accuracy_score(y_test, Y_pred)

acc_gaussian
print(classification_report(y_test, Y_pred))
#perceptron

perceptron = Perceptron()

perceptron.fit(X_train, y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron =100*accuracy_score(y_test, Y_pred)

acc_perceptron
print(classification_report(y_test, Y_pred))
# Decision Tree

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree=100*accuracy_score(y_test, Y_pred)

acc_decision_tree
print(classification_report(y_test, Y_pred))
# Stochastic Gradient Descent

sgd = SGDClassifier()

sgd.fit(X_train, y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = 100*accuracy_score(y_test, Y_pred)

acc_sgd
print(classification_report(y_test, Y_pred))
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(X_test)

acc_random_forest=100*accuracy_score(y_test, Y_pred)

acc_random_forest
print(classification_report(y_test, Y_pred))
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)

acc_log=100*accuracy_score(y_test, Y_pred)

acc_log
print(classification_report(y_test, Y_pred))
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes','Percetron', 

              'Stochastic Gradient Decent', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian,acc_perceptron, 

              acc_sgd,  acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
# Instantiate the classfiers and make a list

classifiers = [svc, 

               knn,

               logreg, 

               random_forest, 

               gaussian,

               decision_tree]



# Define a result table as a DataFrame

result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])



# Train the models and record the results

for cls in classifiers:

    model = cls.fit(X_train, y_train)

    yproba = model.predict_proba(X_test)[::,1]

    

    fpr, tpr, _ = roc_curve(y_test,  yproba)

    auc = roc_auc_score(y_test, yproba)

    

    result_table = result_table.append({'classifiers':cls.__class__.__name__,

                                        'fpr':fpr, 

                                        'tpr':tpr, 

                                        'auc':auc}, ignore_index=True)



# Set name of the classifiers as index labels

result_table.set_index('classifiers', inplace=True)
fig = plt.figure(figsize=(8,6))



for i in result_table.index:

    plt.plot(result_table.loc[i]['fpr'], 

             result_table.loc[i]['tpr'], 

             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

    

plt.plot([0,1], [0,1], color='orange', linestyle='--')



plt.xticks(np.arange(0.0, 1.1, step=0.1))

plt.xlabel("Flase Positive Rate", fontsize=15)



plt.yticks(np.arange(0.0, 1.1, step=0.1))

plt.ylabel("True Positive Rate", fontsize=15)



plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)

plt.legend(prop={'size':13}, loc='lower right')



plt.show()
Predictions = pd.DataFrame({

        "True Value": y_test,

        "Predicted Value": Y_pred

    })

Predictions.head(10)
model = random_forest

model.fit(X,y)

print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.show()
salary_full_data
!pip install BorutaShap
from BorutaShap import BorutaShap



# no model selected default is Random Forest, if classification is True it is a Classification problem

Feature_Selector = BorutaShap(importance_measure='shap', classification=False)



Feature_Selector.fit(X=salary_full_data.iloc[:,0:12], y=salary_full_data.iloc[:,-2], n_trials=100, random_state=0)
np.random.seed(0)

X = salary_full_data.iloc[:,0:12].values

y = salary_full_data.iloc[:,-2].values



f_test, _ = f_regression(X, y)

f_test /= np.max(f_test)



mi = mutual_info_regression(X, y)

mi /= np.max(mi)



plt.figure(figsize=(30, 50))

for i in range(12):

    plt.subplot(3, 4, i + 1)

    plt.scatter(X[:, i], y, edgecolor='black', s=20)

    plt.xlabel("$x_{}$".format(i + 1), fontsize=14)

    if i == 0:

        plt.ylabel("$y$", fontsize=14)

    plt.title("F-test={:.2f}, MI={:.2f}".format(f_test[i], mi[i]),

              fontsize=16)

plt.show()
X = salary_full_data[[ 'ssc_p', 'hsc_p',  'degree_p',  'workex','etest_p', 'specialisation','gender','mba_p','degree_t']]

y = salary_full_data['salary']
scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)
train_X, test_X, train_y,test_y = train_test_split(X_scaled, y, test_size=0.3,random_state=1)
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

my_model.fit(train_X, train_y, verbose=False)
predictions = np.expm1(my_model.predict(test_X))

from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, np.expm1(test_y))))
my_model.fit(X, y)

plot_importance(my_model)

plt.show()