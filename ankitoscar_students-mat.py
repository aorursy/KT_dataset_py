# Import some important tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
# Import .csv file containing data
data_df = pd.read_csv("../input/student-alcohol-consumption/student-mat.csv")
data_df.head()
data_df.shape
data_df.info()
data_df.isna().sum()
data_df.dtypes
data_df.head()
# Function to make colored histograms
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
def color_hist(data,n_bins = 20):
    fig,ax = plt.subplots(tight_layout = True)
    N,bins,patches = ax.hist(data,bins = n_bins)
    fracs = N/N.max()
    norm = colors.Normalize(fracs.min(),fracs.max())
    for thisfrac,thispatch in zip(fracs,patches):    # Normalizing the fraction of data in each bin and giving color to each bin according to that
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
    ax.axvline(x = data.mean(),color='r')  # To show where the mean lies even if the plot looks a bit skewed due to number of bins
# Plotting age of students
color_hist(data_df["age"]);
data_df["age"].mean()
# Relation between grades in all 3 periods and age
fig,ax = plt.subplots(1,3,tight_layout = True,figsize = (18,6))
ax[0].scatter(data_df["age"],data_df["G1"],color = 'r')
ax[0].set(title = "Relation b/w age and grade in 1st period",
          xlabel = "Age",
          ylabel = "1st period grades",
          yticks = np.arange(2,22,3))
ax[1].scatter(data_df["age"],data_df["G2"],color = 'g',marker = '+')
ax[1].set(title = "Relation b/w age and grade in 2nd period",
          xlabel = "Age",
          ylabel = "2nd period grades",
          yticks = np.arange(2,22,3))
ax[2].scatter(data_df["age"],data_df["G3"],marker = 'x')
ax[2].set(title = "Relation b/w age and grade in final period",
          xlabel = "Age",
          ylabel = "Final period grades",
          yticks = np.arange(2,22,3));
obj_cols = []
for column in data_df.columns.tolist():
    if data_df[column].dtype == 'object':
        obj_cols.append(column)
obj_cols
len(obj_cols)
# Plotting all object columns in one go
fig,(ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6,3,figsize = (24,30),tight_layout = True)
for i,obj in enumerate(data_df[obj_cols]):
    if i<3:
        data_df[obj].value_counts().plot(kind = "bar",ax = ax1[i],color = 'indigo').set_title(obj)
    elif i>=3 and i<6:
        data_df[obj].value_counts().plot(kind = "bar",ax = ax2[abs(3-i)]).set_title(obj)
    elif i>=6 and i<9:
        data_df[obj].value_counts().plot(kind = "bar",ax = ax3[abs(6-i)],color = "green").set_title(obj)
    elif i>=9 and i<12:
        data_df[obj].value_counts().plot(kind = "bar",ax = ax4[i-9],color = "orange").set_title(obj)
    elif i>=12 and i<15:
        data_df[obj].value_counts().plot(kind = "bar",ax = ax5[i-12],color = "yellow").set_title(obj)
    else:
        data_df[obj].value_counts().plot(kind = "bar",ax = ax6[i-15],color = "red").set_title(obj)
fig.show();
data_df.head()
# Relation between goout and alcohol consumption w.r.t. final grade
data_df["Alc"] = data_df["Dalc"] + data_df["Walc"]
fig,ax = plt.subplots(tight_layout = True,figsize = (8,6))
ax.scatter(data_df["goout"],data_df["Alc"],c = data_df["G3"],cmap = 'Blues')
ax.set(title = "Relation b/w Going out and Alcohol consumption",
      xlabel = "Going out frequency (1 - very low, 5 - very high)",
      ylabel = "Weekly Alchohol consumption (1 - very low, 10 - very high)",
      xticks = [1,2,3,4,5]);
# Relation between Health and Alcohol consumption w.r.t final period scores
fig,ax = plt.subplots(tight_layout = True,figsize = (8,6))
ax.scatter(data_df["health"],data_df["Alc"],c = data_df["G3"],cmap = "Reds")
ax.set(title = " Relation b/w Health and Alcohol consumption w.r.t. final scores",
       xlabel = "Health(1 - very bad, 5 - very good)",
       ylabel = "Weekly Alcohol consumption(1 - very low, 10 - very high)",
       xticks = [1,2,3,4,5]);
data_df["activities"].replace(['yes','no'],[0,1])
# Plotting relation b/w activities and health w.r.t alcohol consumption
import seaborn as sns
sns.catplot(x = 'health',y = 'activities',kind = 'swarm',hue = 'Alc',data = data_df,height = 6);
# Relation b/w class failures and alcohol consumption w.r.t extra-curricular activities
sns.catplot(x = 'Alc',y = 'failures',hue = 'activities',data = data_df,height = 6)
# Relation b/w higher education pursuit and final grades w.r.t alcohol consumption
sns.catplot(x = 'higher',y = 'G3',hue = 'Alc',data = data_df, height = 6, aspect =11.7/8.27 );
data_df["romantic"].replace(['yes','no'],[1,0])
# Relation b/w alcohol consumption and final grades, w.r.t relationship status
sns.catplot(x = 'Alc',y = 'G3',hue = 'romantic',data = data_df,height = 8.27, aspect = 11.7/8.27);
# Relation b/w grades and studytime w.r.t. extra paid classes 
fig,ax  = plt.subplots(tight_layout = True)
ax.scatter(data_df["studytime"],data_df["G3"],c = data_df["paid"].replace(['yes','no'],[0,1]),cmap = 'YlGn')
ax.set(title = "Relation b/w studytime and grades",
       xlabel = "Weekly Studytime (in hr)",
       ylabel = "Final Grades");
data_df.head()
# Relation between family education support and grades w.r.t school support
fig,ax = plt.subplots(tight_layout = True)
ax.scatter(data_df["famsup"],data_df["G3"],c = data_df["schoolsup"].replace(['yes','no'],[1,0]),cmap = "Blues")
ax.set(title = "Relation b/w family educational support and grades w.r.t school educational support",
       xlabel = "Family Support (yes or no)",
       ylabel = "Final Grades");
# Relation b/w family relation and alcohol consumption w.r.t. grades
fig,ax = plt.subplots(tight_layout = True,figsize = (8,8))
ax.scatter(data_df["famrel"],data_df["Alc"],c = data_df["G3"],cmap = "viridis")
ax.set(title = "Relation b/w family relation and alcohol consumption w.r.t. grades",
       xlabel = "Family Relation ( 1 - very bad, 5 - very good)",
       ylabel = "Weekly Alcohol Consumption( 1 - very low, 10 - very high)")
ax.set_xticks([1,2,3,4,5]);
# Relation between alcohol consumption and grades w.r.t address
fig,ax = plt.subplots(tight_layout = True,figsize = (8,8))
ax.scatter(data_df["Alc"],data_df["G3"],c = data_df["address"].replace(['U','R'],[1,0]), cmap = 'Reds')
ax.set(title = "Relation b/w Alcohol consumption and grades w.r.t. address",
       xlabel = "Alcohol Consumption (1 - very low, 10 - very high)",
       ylabel = "Final Grades");
data_df["G1"].groupby(data_df["sex"]).median()
data_df["G2"].groupby(data_df["sex"]).median()
data_df["G3"].groupby(data_df["sex"]).median()
# Comparing all 3 periods' scores by gender
labels = ['G1','G2','G3']
m_means = [11,11,11]  # Median grades of males and females
w_means = [10,10,10]

x = np.arange(len(labels)) # Location of labels
width = 0.35

fig,ax = plt.subplots(tight_layout = True,figsize = (7,5))
rects1 = ax.bar(x - width/2,m_means,width,label = "Males")
rects2 = ax.bar(x + width/2,w_means,width,label = "Females")

ax.set(title = "Comparision of grades in in all 3 periods by gender",
       ylabel = "Scores",
       xticks = x,
       yticks = np.arange(21),
       xticklabels = labels)
ax.legend();
# Comparing grades in all 3 periods based on gender w.r.t. alcohol consumption
sns.catplot(x = 'Alc',y = 'G1',hue = 'sex', data = data_df,orient='v')
sns.catplot(x = 'Alc',y = 'G2',hue = 'sex', data = data_df,orient='v')
sns.catplot(x = 'Alc',y = 'G3',hue = 'sex', data = data_df,orient='v');
# Relation between freetime and workday alcohol consumption, going out and weekend alcohol consumption w.r.t gender
fig,ax = plt.subplots(1,2,tight_layout = True,figsize = (14,7))
ax[0].scatter(data_df["freetime"],data_df["Dalc"],c = data_df["sex"].replace(['M','F'],[1,0]),cmap = 'viridis')
ax[0].set(title = "Relation b/w freetime and workday alcohol consumption w.r.t. gender",
          xlabel = "Freetime after school ( 1- very low, 5 - very high)",
          ylabel = "Workday Alcohol Consumption ( 1 - very low, 5 - very high)",
          xticks = np.arange(6),
          yticks = np.arange(6))
ax[1].scatter(data_df["goout"],data_df["Walc"],c = data_df["sex"].replace(['M','F'],[1,0]),cmap = 'plasma')
ax[1].set(title = "Relation b/w going out and weekend alcohol consumption w.r.t. gender",
          xlabel = "Going out (1 - very low, 5 - very high)",
          ylabel = "Weekend Alcohol Consumption (1 - very low, 5 - very high)",
          xticks = np.arange(6),
          yticks = np.arange(6));
# Relation between scores in each period w.r.t. internet availability
ls = ['G1','G2','G3']
for i in range(3):
    sns.catplot(x = 'internet',y = ls[i],data = data_df);
# Relation between grades and mother's and father's education w.r.t. mother's and father's education
sns.catplot(x = 'Medu',y = 'G3',data = data_df,hue = 'Mjob',aspect = 11.7/8.27,height = 5)
sns.catplot(x = 'Fedu',y = 'G3',data = data_df,hue = 'Fjob',aspect = 11.7/8.27,height = 5);
# Relation between absences and final grade w.r.t going out, freetime and weekly alcohol consumption
ls = ["goout","freetime","Alc"]
fig,ax = plt.subplots(1,3,tight_layout = True,figsize = (15,9))
for i in range(3):
    ax[i].scatter(data_df['absences'],data_df["G3"],c = data_df[ls[i]],cmap = 'viridis')
    head = "Relation between absences and final grade in "+ls[i]
    ax[i].set(title = head,
              xlabel = "Absences",
              ylabel = "Final Grade");
# Relation between parent's cohabitation situation and grades
sns.catplot(x = "Pstatus",y = "G3",data = data_df,height = 6,aspect = 11.7/8.27);
# Splitting data into features and labels
X = data_df.drop('G3',axis = 1)
y = data_df["G3"]
X.head(),y.head()
# Label encoding our data
# Function for label encoding data
from sklearn.preprocessing import LabelEncoder
def label_encoding(data):
    for column in obj_cols:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
    return data
X1 = label_encoding(X)
X1.head()
X = X1
X.head()
# Splitting data into train and test set
np.random.seed(42)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
X_train.shape,y_train.shape
X_train.head()
# Making a evaluation function
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
def evaluate(model,X_train,y_train,X_test,y_test):
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    scores = {
        "Training R^2 score" : r2_score(y_train,train_preds),
        "Test R^2 score" : r2_score(y_test,test_preds),
        "Training MAE" : mean_absolute_error(y_train,train_preds),
        "Test MAE" : mean_absolute_error(y_test,test_preds),
        "Training MSE" : mean_squared_error(y_train,train_preds),
        "Test MSE" : mean_squared_error(y_test,test_preds)
    }
    print(scores)
    scores_df = pd.DataFrame(scores,index=[0])
    scores_df.T.plot.bar(color = "salmon");
    return scores
# Using RidgeRegression() model
from sklearn.linear_model import Ridge
model = Ridge(alpha=0.5)
model.fit(X_train,y_train)
ridge = evaluate(model,X_train,y_train,X_test,y_test)
# Function to train a model and evaluate it
def train_model(model,X_train,y_train,X_test,y_test):
    md = model
    md.fit(X_train,y_train)
    evaluate(md,X_train,y_train,X_test,y_test)
    return md
# Using support vector regression model in linear kernel
from sklearn.svm import SVR
linear = train_model(SVR(kernel='linear'),X_train,y_train,X_test,y_test)
# Training using SVR model using 'rbf' kernel
rbf = train_model(SVR(kernel = 'rbf'),X_train,y_train,X_test,y_test)
# Train using RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
forest = train_model(RandomForestRegressor(),X_train,y_train,X_test,y_test)
X = X.drop('famsup',axis = 1)
np.random.seed(42)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
train_model(RandomForestRegressor(),X_train,y_train,X_test,y_test)
X = X.drop('schoolsup',axis = 1)
np.random.seed(42)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
train_model(RandomForestRegressor(),X_train,y_train,X_test,y_test)
X = X.drop('goout',axis = 1)
np.random.seed(42)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
train_model(RandomForestRegressor(),X_train,y_train,X_test,y_test)
X = X.drop('Pstatus',axis = 1)
np.random.seed(42)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
train_model(RandomForestRegressor(),X_train,y_train,X_test,y_test)
X["absences"].groupby(X["absences"]>=40).count()
#X.drop(X[X.absences>=40].index,inplace = True)
np.random.seed(42)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
train_model(RandomForestRegressor(),X_train,y_train,X_test,y_test)
X = X.drop('Alc',axis = 1)
np.random.seed(42)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
train_model(RandomForestRegressor(),X_train,y_train,X_test,y_test)
# Please comment this cell while running all cells, run this cell after all cells are run
#X1 = X
#X1["Alc"] = X["Dalc"]+X["Walc"]
#X1.drop("Dalc",axis = 1, inplace = True)
#X1.drop("Walc",axis = 1,inplace = True)
#np.random.seed(42)
#X1_t,X1_ts,y_t,y_ts = train_test_split(X1,y,test_size = 0.2)
#train_model(RandomForestRegressor(),X1_t,y_t,X1_ts,y_ts)
X["Alc"] = X["Dalc"]+X["Walc"]
X.head()
# Cross validating on all data for R^2,MAE and MSE
from sklearn.model_selection import cross_val_score
np.random.seed(42)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
model = train_model(RandomForestRegressor(),X_train,y_train,X_test,y_test)
cross_val_score(model,X,y,cv = 5).mean()
import sklearn
sorted(sklearn.metrics.SCORERS.keys())
cross_val_score(model,X,y,cv=5,scoring ='neg_mean_absolute_error').mean()
cross_val_score(model,X,y,cv=5,scoring ='neg_mean_squared_error').mean()
cross_val_scores = {
    'R^2 score' : 0.8265673076826653,
    'Mean Absolute Error' : 1.1075949367088607,
    'Mean Squared Error' : 3.342240253164557
}
cross_val_scores = pd.DataFrame(cross_val_scores,index = [0])
cross_val_scores.T.plot.bar(color = "lightblue");
# Feature Importance of final model
feat_importances = pd.Series(model.feature_importances_,index = X.columns)
feat_importances.nlargest(9).plot.barh(color = 'indigo',figsize=(10,10));
# Exporting the model using joblib
import joblib
joblib.dump(model,"model.joblib")
