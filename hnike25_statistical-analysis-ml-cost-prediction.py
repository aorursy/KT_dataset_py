import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
sns.set_style('ticks')
df = pd.read_csv('../input/insurance.csv')
f,ax = plt.subplots(figsize=(10,5))
sns.countplot(x='region', data=df, palette="hls",orient='v',ax=ax,edgecolor='0.2')
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()+0.3, i.get_height()+3, \
            str(round((i.get_height()/df.region.shape[0])*100))+'%', fontsize=12,
                color='black') 
ax.set_xlabel("Region", fontsize=13)
ax.tick_params(length=3,labelsize=12,labelcolor='black')
ax.set_title("Region Distribution", fontsize=14)
x_axis = ax.axes.get_yaxis().set_visible(False)
sns.despine(left=True)
plt.show()
#Let classify age into 4 well known categories, which are 
#'Adolescent',"Young Adult","Adult","Senior"
cut_points = [17,20,35,50,65]
label_names = ['Adolescent',"Young Adult","Adult","Senior"]
df["age_cat"] = pd.cut(df["age"],cut_points,labels=label_names)

f,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
sns.countplot(x='age_cat',data=df,palette='Greens_r',orient='v',ax=ax1,edgecolor='0.2')
for i in ax1.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax1.text(i.get_x()+0.3, i.get_height()+3, \
            str(round((i.get_height()/df.age_cat.shape[0])*100))+'%', fontsize=12,
                color='black') 
ax2.hist('age', bins=10,data=df,edgecolor='0.2')
ax1.set_xlabel("Age Categories", fontsize=13)
ax1.tick_params(length=3,labelsize=12,labelcolor='black')
ax1.set_title("Age Distribution by Categories", fontsize=14)
ax2.set_xlabel('Age',fontsize=13)
ax2.set_title('Age Distribution',fontsize=14)
x_axis = ax1.axes.get_yaxis().set_visible(False)

f.subplots_adjust(wspace=0.22,right=1.5)
sns.despine(left=True)
plt.show()
def gender_dist_plot(x_val,title):
    f,ax = plt.subplots(figsize=(10,5))
    sns.countplot(x=x_val, data=df, palette=['dodgerblue','lightpink'],hue='sex',hue_order=['male','female'],
                  orient='v',ax=ax,edgecolor='0.2')
    for i in ax.patches:
        ax.text(i.get_x()+0.1, i.get_height()+3, \
                str(round((i.get_height()/df.region.shape[0])*100))+'%', fontsize=11,
                    color='black') 
    ax.set_xlabel(title, fontsize=12,color='black')
    ax.tick_params(length=3,labelsize=12,labelcolor='black')
    ax.set_title(title +' Distribution by Gender', fontsize=13)
    x_axis = ax.axes.get_yaxis().set_visible(False)
    ax.legend(loc=[1,0.9],fontsize=12,title='Gender Type',ncol=2)
    sns.despine(left=True)
    return plt.show()

gender_dist_plot('age_cat','Age Category')
gender_dist_plot('region','Region')
## Region Distribution by Male Smoker
male_data = df[df.sex=='male']
female_data = df[df.sex=='female']

def sex_dist(data,gender,title_color):    
    f,ax = plt.subplots(figsize=(10,5))
    sns.countplot(x='region', data=data, palette=['ForestGreen','saddlebrown'],hue='smoker',
                  hue_order=['no','yes'],orient='v',ax=ax,edgecolor='0.2')
    for i in ax.patches:
        # get_x pulls left or right; get_height pushes up or down
        ax.text(i.get_x()+0.1, i.get_height()+3, \
                str(round((i.get_height()/data.region.shape[0])*100))+'%', fontsize=12,
                    color='black') 
    ax.set_xlabel("Region", fontsize=13)
    ax.tick_params(length=3,labelsize=12,labelcolor='black')
    ax.set_title('Region Distribution by '+ gender +' Smoker', fontsize=14,color=title_color)
    x_axis = ax.axes.get_yaxis().set_visible(False)
    ax.legend(loc=[1,0.9],fontsize=12,title='Smoker type')
    sns.despine(left=True)
    return plt.show()

sex_dist(male_data,'Male','blue')
sex_dist(female_data,'Female','purple')
from scipy import stats
from scipy.stats import norm, skew,kurtosis

def data_transform(data,input):
    f,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(8,8))
    #plots
    sns.boxplot(x=input, data = data,ax=ax1,orient='v')
    sns.distplot(data[input],ax=ax2,color='blue',hist=False)
    res = stats.probplot(data[input], plot=ax3)

    axes = [ax1,ax2]
    kwargs = {'fontsize':14,'color':'black'}
    #for i in range(len(axes)):
        #x_axis = axes[i].axes.get_yaxis().set_visible(False)
    ax1.set_title(input+' Boxplot Analysis',**kwargs)
    ax1.set_xlabel('Box',**kwargs)
    ax1.set_ylabel('BMI Values',**kwargs)

    ax2.set_title(input+' Distribution',**kwargs)
    ax2.set_xlabel(input+' values',**kwargs)

    ax3.set_title('Probability Plot',**kwargs)
    ax3.set_xlabel('Theoretical Quantiles',**kwargs)
    ax3.set_ylabel('Ordered Values',**kwargs)

    f.subplots_adjust(wspace=0.22,right= 2)
    sns.despine()
    return plt.show()

data_transform(df,'bmi')
cut_points = [14,19,25,30,65]
label_names = ['Underweight',"normal","overweight","obese"]
df["bmi_cat"] = pd.cut(df['bmi'],cut_points,labels=label_names)
gender_dist_plot('bmi_cat','BMI')
data_transform(df,'charges')
df.charges = np.log1p(df.charges)
data_transform(df,'charges')
def scatter_analysis(hue_type,palette,data):
    sns.lmplot(x = 'bmi',y='charges',hue=hue_type,data=data,palette=palette,size=6,aspect=1.5,
           scatter_kws={"s": 70, "alpha": 1,'edgecolor':'black'},legend=False,fit_reg=True)
    plt.title('Scatterplot Analysis',fontsize=14)
    plt.xlabel('BMI',fontsize=12)
    plt.ylabel('Charge',fontsize=12)
    plt.legend(loc=[1.1,0.5],title = hue_type, fontsize=13)
plt.show()
scatter_analysis('smoker',['ForestGreen','saddlebrown'],df)
plt.figure(figsize=(12,8))
kwargs = {'fontsize':12,'color':'black'}
sns.heatmap(df.corr(),annot=True,robust=True)
plt.title('Correlation Analysis on the Dataset',**kwargs)
plt.tick_params(length=3,labelsize=12,color='black')
plt.yticks(rotation=0)
plt.show()
#Let drop all categorical variable create during the EDA Analysis
df.drop(['age_cat','bmi_cat'],axis=1,inplace=True)
##Split the data into smoker dataset and non-smoker dataset
df_smoker = df[df.smoker=='yes']
# Convert all categorical columns in the dataset to Numerical for the Analysis
df_smoker = pd.get_dummies(df_smoker,drop_first=True)
from scipy.stats import pearsonr
plt.figure(figsize=(12,8))
kwargs = {'fontsize':12,'color':'black'}
sns.heatmap(df_smoker.corr(),annot=True,robust=True)
plt.title('Correlation Analysis for Smoker',**kwargs)
plt.tick_params(length=3,labelsize=12,color='black')
plt.yticks(rotation=0)
plt.show()
#p_value Analysis
p_value = [round(pearsonr(df_smoker['charges'],df_smoker[i])[1],4) for i in df_smoker.columns]
pvalue_table = pd.DataFrame(p_value,df_smoker.columns).reset_index()
pvalue_table.columns=['colmuns_name','p_value']
pvalue_table.sort_values('p_value')
df_smoker.drop(['children','sex_male', 'region_northwest',
       'region_southeast', 'region_southwest'],axis=1,inplace=True)
scatter_analysis(None,['ForestGreen','saddlebrown'],df_smoker)
from sklearn.metrics import explained_variance_score,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
X = df_smoker.drop('charges',axis=1)
y = df_smoker['charges']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
#Standardizing the values
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
## Build  & Evaluate our Model
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('intercept: {:.4f} \ncte1: {:.4f} \ncte2: {:.4f}'.format(model.intercept_,model.coef_[0],model.coef_[1]))

print('Model_Accuracy_Score (R Square): {:.4f} \nLoss(RMSE): {:.4f}'.format(r2_score(y_pred,y_test),np.sqrt(mean_squared_error(y_pred,y_test))))
def model_scatter_plot(model):
    title = str(model)
    title = title.split('.')[3]
    title = title.replace("'>",'')
    lreg = model()
    lreg.fit(X_train,y_train)
    y_pred = lreg.predict(X_test)
    #model_table
    model_table = pd.DataFrame(y_pred,y_test).reset_index()
    model_table.columns=['y_test','y_pred']
    #Model Graph
    sns.lmplot(x = 'y_test',y='y_pred',data = model_table,size=6,aspect=1.5,
           scatter_kws={"s": 70, "alpha": 1,'edgecolor':'black'},fit_reg=True)
    plt.title(title +' Analysis',fontsize=14)
    plt.xlabel('y_test',fontsize=12)
    plt.ylabel('y_pred',fontsize=12)
    #plt.scatter(y_test,y_pred)
    return plt.show()

model_scatter_plot(LinearRegression)
def model_apply(age,bmi_value):
    ## Exple: for a smoker who is age number with bmi = bmi_value, 
    #how much would he pay for insurance
    c = [[age, bmi_value]]
    #we have to transform the data from the standard scaler
    c = sc.transform(c)
    charge_value = model.coef_[0]*(c[0][0]) + model.coef_[1]*(c[0][1]) + model.intercept_
    charge_value = np.exp(charge_value) 
    x = ('The Insurrance Charges for a {:.1f} years old person who is a Smoker with an bmi = {:.1f} will be {:.4f}'.format(age,bmi_value,charge_value))
    # we use the np.exp() because we transformed the value of charge during the charge EDA earlier above
    return print(x)
#if you are a smoker of 19 yr old and bmi of 32 then what insurrance would you be charged?
model_apply(19,32)
def robust_model(input):
    #Model type to evaluate
    model_list = [ExtraTreesRegressor(),RandomForestRegressor(),GradientBoostingRegressor(),
            LinearRegression(),xgb.XGBRegressor()]
    r_score = []
    loss = []
    for reg in model_list:
        reg.fit(X_train,y_train)
        y_pred = reg.predict(X_test)
        r_score.append(explained_variance_score(y_pred,y_test))
        loss.append(np.sqrt(mean_squared_error(y_pred,y_test)))
    ## Model score table
    model_str = ['ExtraTrees','Random Forest','Gradient Boosting',
            'Linear Regression','XGB Regressor']
    other_model = pd.DataFrame(r_score,model_str).reset_index()
    other_model.columns = ['Model','R(Square)']
    other_model['loss'] = loss
    other_model.sort_values('R(Square)',ascending=False,inplace=True)
    ## Model Graph
    ax = other_model[['R(Square)','loss']].plot(kind='bar',width=0.7,
                            figsize=(15,7), color=['slategray', 'darkred'], fontsize=13,edgecolor='0.2')
    for i in ax.patches:
        # get_x pulls left or right; get_height pushes up or down
        ax.text(i.get_x()+.1, i.get_height()+0.01, \
                str(round((i.get_height()), 3)), fontsize=12, color='black',)
    ax.set_title('Regression Model Evaluation For '+input,fontsize=14,color='black')
    ax.set_xticklabels(other_model.Model, rotation=0, fontsize=12)
    ax.set_xlabel('Model',**kwargs)
    x_axis = ax.axes.get_yaxis().set_visible(False)
    sns.despine(left=True)
    return plt.show()

robust_model('Smoker')
# Let's visualize the test data vs. the predicted data
model_scatter_plot(ExtraTreesRegressor)

model_scatter_plot(GradientBoostingRegressor)

model_scatter_plot(RandomForestRegressor)
df_non_smoker = df[df.smoker=='no']

scatter_analysis(None,['ForestGreen','saddlebrown'],df_non_smoker)
# Convert all categorical columns in the dataset to Numerical for the Analysis
df_non_smoker['children'] = df_non_smoker['children'].astype('category')
df_non_smoker = pd.get_dummies(df_non_smoker,drop_first=True)
#correlation Analysis
plt.figure(figsize=(12,8))
kwargs = {'fontsize':12,'color':'black'}
sns.heatmap(df_non_smoker.corr(),annot=True,robust=True)
plt.title('Correlation Analysis for Smoker',**kwargs)
plt.tick_params(length=3,labelsize=12,color='black')
plt.yticks(rotation=0)
plt.show()
# Let plot the age vs. charge scatter plot to see the correlation between them
sns.lmplot(x = 'age',y='charges',data=df_non_smoker,size=6,aspect=1.5,
           scatter_kws={"s": 70, "alpha": 1,'edgecolor':'black'},legend=False,fit_reg=True)
plt.title('Scatterplot Analysis',fontsize=14)
plt.show()
X = df_non_smoker['age']
y = df_non_smoker['charges']
X = X.reshape(-1, 1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
#Standardizing the values
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#Model fitting
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print('intercept: {:.4f} \ncte1: {:.4f}'.format(model.intercept_,model.coef_[0]))

print('Model_Accuracy_Score (R Square): {:.4f} \nLoss(RMSE): {:.4f}'.format(r2_score(y_pred,y_test),np.sqrt(mean_squared_error(y_pred,y_test))))
model_scatter_plot(LinearRegression)
robust_model('Non Smoker')
