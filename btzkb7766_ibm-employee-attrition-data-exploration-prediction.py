import pandas as pd 
import numpy as np 
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
pd.set_option('display.max_columns', None) # Show max columns of dataset 
# Columns and their content
df.nunique()
#Total column and rows
df.shape
# Data type 
df.info()
# Data description 
df.describe()
# Number of null values 
df.isnull().sum()
#Number of duplicate values
df.duplicated().sum()
# Max values of the data
df.max()
# Removing unuseful data
df.drop(['Over18','EmployeeCount','StandardHours'], axis = 1, inplace= True)
# Changing the Attrition value to numerical 
df['Attrition']= df['Attrition'].replace({"Yes": 1, "No": 0,}) 
# Splitting data for comparision (employees who left and stayed)
left = df[df['Attrition'] == 1] # Employees who left 
stayed = df[df['Attrition'] == 0] # Employees who stayed

# Job info
jobsinfo_left = left[['EmployeeNumber','JobLevel', 'JobInvolvement', 'Department','JobRole','JobSatisfaction','PerformanceRating']]
jobsinfo_stayed = stayed[['EmployeeNumber','JobLevel', 'JobInvolvement', 'Department','JobRole','JobSatisfaction','PerformanceRating']]

# Travel
travel_left = left[['EmployeeNumber','BusinessTravel','DistanceFromHome']]
travel_stayed = stayed[['EmployeeNumber','BusinessTravel','DistanceFromHome']]

# Money
money_left = left[['EmployeeNumber','DailyRate','HourlyRate','MonthlyRate','MonthlyIncome','PercentSalaryHike']]
money_stayed = stayed[['EmployeeNumber','DailyRate','HourlyRate','MonthlyRate','MonthlyIncome','PercentSalaryHike']]

# Expereince/ Education
eduexp_left = left[['EmployeeNumber','Education','EducationField','NumCompaniesWorked','TotalWorkingYears', 'TrainingTimesLastYear']]
eduexp_stayed = stayed[['EmployeeNumber','Education','EducationField','NumCompaniesWorked','TotalWorkingYears', 'TrainingTimesLastYear']]

# Emotions
emotions_left = left[['EmployeeNumber','EnvironmentSatisfaction','OverTime','RelationshipSatisfaction']]
emotions_stayed = stayed[['EmployeeNumber','EnvironmentSatisfaction','OverTime','RelationshipSatisfaction']]

# Time
time_left =left[['EmployeeNumber','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']]
time_stayed =stayed[['EmployeeNumber','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']]

# Additional Info
additionalinfo_left = left[['EmployeeNumber','Age','Gender','MaritalStatus','WorkLifeBalance']]
additionalinfo_stayed= stayed[['EmployeeNumber','Age','Gender','MaritalStatus','WorkLifeBalance']]
# Amount of people who left and stayed 
pd.DataFrame(([[left['Attrition'].count(),stayed['Attrition'].count()]]),index= ['Number of people'], columns= ['People who left','People who stayed'])
jobinfo_benchmark = round(pd.DataFrame(df[['JobInvolvement', 'Department','JobRole','JobSatisfaction','PerformanceRating']].mean(),columns=['Benchmark Score']),ndigits=2)
travelinfo_benchmark = round(pd.DataFrame(df[['DistanceFromHome']].mean(),columns=['Benchmark Score']),ndigits=2)
moneyinfo_benchmark = round(pd.DataFrame(df[['DailyRate','HourlyRate','MonthlyRate','MonthlyIncome','PercentSalaryHike']].mean(),columns=['Benchmark Score']),ndigits=2)
eduexpinfo_benchmark = round(pd.DataFrame(df[['Education','NumCompaniesWorked','TotalWorkingYears', 'TrainingTimesLastYear']].mean(),columns=['Benchmark Score']),ndigits=2)
emotion_benchmark = round(pd.DataFrame(df[['EnvironmentSatisfaction','OverTime','RelationshipSatisfaction']].mean(),columns=['Benchmark Score']),ndigits=2)
time_benchmark = round(pd.DataFrame(df[['YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']].mean(),columns=['Benchmark Score']),ndigits=2)
# Comparing average score to benchmark
ass = round(pd.DataFrame(jobsinfo_stayed.mean(),columns=['Average Score Stayed']),ndigits = 2).iloc[2:,:]
asl = round(pd.DataFrame(jobsinfo_left.mean(),columns=['Average Score Left']),ndigits = 2).iloc[2:,:]
pd.concat([ass,asl,jobinfo_benchmark], axis=1).iloc[:,:]
# Calculation to number of people who are above or below the benchmark
jil = (jobsinfo_left['JobInvolvement']>= 2.37).value_counts()
jis =(jobsinfo_stayed['JobInvolvement']>= 2.37).value_counts()
jsl =(jobsinfo_left['JobSatisfaction']>= 2.37).value_counts()
jss =(jobsinfo_stayed['JobSatisfaction']>= 2.37).value_counts()
prl =(jobsinfo_left['PerformanceRating']>= 3.15).value_counts()
prs =(jobsinfo_stayed['PerformanceRating']>= 3.15).value_counts()
# Manual list creation for dataframe 
job_people_above_benchamrk = [138,874,125,776,37,189]
job_people_below_benchamrk = [99,359,112,457,200,1044]
pd.DataFrame([job_people_above_benchamrk,job_people_below_benchamrk],index=['Number of people above benchmark','Number of people below benchmark']
             ,columns=['Job Involvment~left','Job Involvment~stayed','Job Satisfaction~left'
                       ,'Job Satisfaction~stayed','Performance Rating~left','Performance Rating~stayed'])
# The amount of people that worked in each deparment and their job role
departmentl =left['Department'].value_counts()
departmetns = stayed['Department'].value_counts()
pd.DataFrame([departmentl,departmetns],index=['People who left','People who stayed'])
# Job roles of people who left and stayed
jobrolel =left['JobRole'].value_counts()
jobroles = stayed['JobRole'].value_counts()
pd.DataFrame([jobrolel,jobroles],index=['People who left','People who stayed'])
tras = round(pd.DataFrame(travel_stayed.mean(),columns=['Average Score Stayed']),ndigits = 2).iloc[1:,:]
tral = round(pd.DataFrame(travel_left.mean(),columns=['Average Score Left']),ndigits = 2).iloc[1:,:]
pd.concat([tras,tral,travelinfo_benchmark], axis=1).iloc[:,:]
btl =travel_left['BusinessTravel'].value_counts()
bts = travel_stayed['BusinessTravel'].value_counts()
pd.DataFrame([btl,bts],index=['People who left','People who stayed'])
# Comparing average score to benchmark
ms = round(pd.DataFrame(money_stayed.mean(),columns=['Average Rate Stayed']),ndigits = 2)
ml = round(pd.DataFrame(money_left.mean(),columns=['Average Rate Left']),ndigits = 2)
pd.concat([ms,ml,moneyinfo_benchmark], axis=1).dropna()
drl = (money_left['DailyRate']>= 802.49).value_counts()
drs =(money_stayed['DailyRate']>=802.49).value_counts()
hrl =(money_left['HourlyRate']>= 65.89).value_counts()
hrs =(money_stayed['HourlyRate']>= 65.89).value_counts()
mil =(money_left['MonthlyIncome']>= 6502.93).value_counts()
mis =(money_stayed['MonthlyIncome']>= 6502.93).value_counts()
mrl =(money_left['MonthlyRate']>= 14313.10).value_counts()
mrs =(money_stayed['MonthlyRate']>= 14313.10).value_counts()
pshl =(money_left['PercentSalaryHike']>= 15.21).value_counts()
pshs =(money_stayed['PercentSalaryHike']>= 15.21).value_counts()
# Manual list creation for dataframe 
money_people_above_benchamrk = [104,630,119,627,52,441,122,608,87,464]
money_people_below_benchamrk = [133,603,118,606,185,792,115,625,150,769]
money_benchmark = [802.49,802.49,65.89,65.89,6502.93,6502.93,14313.10,14313.10,15.21,15.21]
pd.DataFrame([money_people_above_benchamrk,money_people_below_benchamrk,money_benchmark],
             index=['Number of people above benchmark','Number of people below benchmark','Benchmark']
             ,columns=['DailyRate~left','DailyRate~stayed','HourlyRate~left'
                       ,'HourlyRate~stayed','MonthlyIncome~left','MonthlyIncome~stayed',
                      'MonthlyRate~left','MonthlyRate~stayed','PercentSalaryHike~left','PercentSalaryHike~stayed'])
# Calculating the mean of each job role
combine_jrmi = df[['JobRole', 'MonthlyIncome']]
aise = combine_jrmi[combine_jrmi['JobRole']=='Sales Executive'].mean()
airs = combine_jrmi[combine_jrmi['JobRole']=='Research Scientist'].mean()
ailt = combine_jrmi[combine_jrmi['JobRole']=='Laboratory Technician'].mean()
aimd = combine_jrmi[combine_jrmi['JobRole']=='Manufacturing Director'].mean()
aihr = combine_jrmi[combine_jrmi['JobRole']=='Healthcare Representative'].mean()
aim = combine_jrmi[combine_jrmi['JobRole']=='Manager'].mean()
aisr = combine_jrmi[combine_jrmi['JobRole']=='Sales Representative'].mean()
aird = combine_jrmi[combine_jrmi['JobRole']=='Research Director'].mean()
aihur = combine_jrmi[combine_jrmi['JobRole']=='Human Resources'].mean()

# Creating the dataset for average income based on job roles 
avg_income =pd.DataFrame([aise,airs,ailt,aimd,
                          aihr,aim,aisr,aird,aihur],index= ['Sales Executive','Research Scientist','Laboratory Technician',
                                                                   'Manufacturing Director','Healthcare Representative',
                                                                   'Manager','Sales Representative',
                                                                   'Research Director','Human Resources'], columns = ['MonthlyIncome']).round(decimals=2)

# Renaming the column
avg_income.columns = [c.replace('MonthlyIncome', 'Average Monthly Income') for c in avg_income.columns]
avg_income
combine_jrmi.max()
df[df['MonthlyIncome'] == 19999]
# Comparing average score to benchmark
eds = round(pd.DataFrame(eduexp_stayed.mean(),columns=['Average Score Stayed']),ndigits = 2).iloc[1:,:]
edl = round(pd.DataFrame(eduexp_left.mean(),columns=['Average Score Left']),ndigits = 2).iloc[1:,:]
pd.concat([eds,edl,eduexpinfo_benchmark], axis=1)
# Calculation to number of people who are above or below the benchmark
el = (eduexp_left['Education']>= 2.91).value_counts()
es =(eduexp_stayed['Education']>= 2.91).value_counts()
ncwl =(eduexp_left['NumCompaniesWorked']>= 2.69).value_counts()
ncws =(eduexp_stayed['NumCompaniesWorked']>= 2.69).value_counts()
twyl =(eduexp_left['TotalWorkingYears']>= 11.28).value_counts()
twys =(eduexp_stayed['TotalWorkingYears']>= 11.28).value_counts()
ttlyl =(eduexp_left['TrainingTimesLastYear']>= 2.8).value_counts()
ttlys =(eduexp_stayed['TrainingTimesLastYear']>= 2.8).value_counts()
# Manual list creation for dataframe 
edu_people_above_benchamrk = [162,856,100,506,48,463,115,683]
edu_people_below_benchamrk = [75,377,137,727,189,770,122,550]
pd.DataFrame([edu_people_above_benchamrk,edu_people_below_benchamrk],index=['Number of people above benchmark','Number of people below benchmark']
             ,columns=['Education for people who left','Education for people who stayed',
                                                           'Companies Worked~left','Companies Worked~stayed','Work Exepereince~left','Work Exepereince~stayed',
                                                           'Training times for last year~left','Training times for last year~stayed'])
df['TotalWorkingYears'].max()
df[df['TotalWorkingYears'] == 40]
# Comparing average score to benchmark
emos = round(pd.DataFrame(emotions_stayed.mean(),columns=['Average Score Stayed']),ndigits = 2).iloc[1:,:]
emol = round(pd.DataFrame(emotions_left.mean(),columns=['Average Score Left']),ndigits = 2).iloc[1:,:]
pd.concat([emos,emol,emotion_benchmark], axis=1)
# Calculation to number of people who are above or below the benchmark
esl = (emotions_left['EnvironmentSatisfaction']>= 2.46).value_counts()
ess =(emotions_stayed['EnvironmentSatisfaction']>= 2.77).value_counts()
resl =(emotions_left['RelationshipSatisfaction']>= 2.60).value_counts()
ress =(emotions_stayed['RelationshipSatisfaction']>= 2.73).value_counts()

pd.DataFrame([esl,ess,resl,ress])
# Manual list creation for dataframe 
emotion_people_above_benchamrk = [122,777,135,756]
emotion_people_below_benchamrk = [115,456,102,477]
pd.DataFrame([emotion_people_above_benchamrk,emotion_people_below_benchamrk],index=['Number of people above benchmark','Number of people below benchmark']
             ,columns=['Environment Satisfaction~left','Environment Satisfaction~stayed','Relationship Satisfaction~left'
                       ,'Relationship Satisfaction~stayed'])
# Comparing average score to benchmark
ts = round(pd.DataFrame(time_stayed.mean(),columns=['Average Years Stayed']),ndigits = 2).iloc[1:,:]
tl = round(pd.DataFrame(time_left.mean(),columns=['Average Years Left']),ndigits = 2).iloc[1:,:]
pd.concat([ts,tl,time_benchmark], axis=1)
# Calculation to number of people who are above or below the benchmark
yacl = (time_left['YearsAtCompany']>= 7.01).value_counts()
yacs =(time_stayed['YearsAtCompany']>= 7.01).value_counts()
yicrl =(time_left['YearsInCurrentRole']>= 4.23).value_counts()
yicrs =(time_stayed['YearsInCurrentRole']>= 4.23).value_counts()
yslpl = (time_left['YearsSinceLastPromotion']>= 2.19).value_counts()
yslps =(time_stayed['YearsSinceLastPromotion']>= 2.19).value_counts()
ywcml =(time_left['YearsWithCurrManager']>= 4.12).value_counts()
ywcms =(time_stayed['YearsWithCurrManager']>= 4.12).value_counts()

pd.DataFrame([yacl,yacs,yicrl,yicrs,yslpl,yslps,ywcml,ywcms])
# Manual list creation for dataframe 
time_people_above_benchamrk = [55,473,54,504,51,322,61,486]
time_people_below_benchamrk = [182,760,183,729,186,911,176,747]
pd.DataFrame([time_people_above_benchamrk,time_people_below_benchamrk],index=['Number of people above average years','Number of people below average years']
             ,columns=['Years At Company~left','Years At Company~stayed','Years In Current Role~left'
                       ,'Years In Current Role~stayed','Years Since Last Promotion~left','Years Since Last Promotion',
                       'Years With Current Manager~left'
                       ,'Years With Current Manager~stayed'])
# Total number of people who worked over time 
df['OverTime'].value_counts()
# Percentage of people who worked overtime
yes = 416
no = 1054
total = yes+no
percentage = yes/total *100
round(percentage, ndigits=2)
# df.groupby(['YearsAtCompany']).sort_values([df],ascending = False)
stayed.sort_values(['YearsAtCompany'],ascending = False).head(10)
additionalinfo_left['Gender'].value_counts()
additionalinfo_left['Age'].min()
additionalinfo_left['Age'].max()
additionalinfo_left['MaritalStatus'].value_counts()
# Mean calculation
wb0 = left['WorkLifeBalance'].mean()
ji0 = left['JobInvolvement'].mean()
js0 = left['JobSatisfaction'].mean()
mi0 = left['MonthlyIncome'].mean()
rs0 = left['RelationshipSatisfaction'].mean()
print(wb0,ji0,js0,mi0,rs0)
# Predicted employee attrition  
stayed.loc[(df['WorkLifeBalance'] <= 2.66) & (df['JobInvolvement'] <= 2.52)& (df['JobSatisfaction'] <= 2.4) & (df['MonthlyIncome'] <= 4787) & (df['RelationshipSatisfaction'] <= 2.6)] 
X = df[['WorkLifeBalance','JobInvolvement','JobSatisfaction','MonthlyIncome','RelationshipSatisfaction']] 
y = df['Attrition']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train.values.reshape(-1,1))
from sklearn.linear_model import LogisticRegression
classifierlr = LogisticRegression(random_state = 0)
classifierlr.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifierlr.predict(X_test)

#Accuracy 
lr_accuracy = round(classifierlr.score(X_train, y_train) * 100, 2)
lr_accuracy
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifierlr, X = X_train, y = y_train, cv = 10)
lrk = round(accuracies.mean() *100,ndigits=2)
lrk
from sklearn.neighbors import KNeighborsClassifier
classifierknn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) #choosing the metric to select the distance. P = 2 means the distance selected is the euclidean distance
classifierknn.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifierknn.predict(X_test)

#Accuracy 
knn_accuracy = round(classifierknn.score(X_train, y_train) * 100, 2)
knn_accuracy
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifierknn, X = X_train, y = y_train, cv = 10)
knnk = round(accuracies.mean() *100,ndigits=2)
knnk
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifiersvc = SVC(kernel = 'linear', random_state = 0) #choose the kernel
classifiersvc.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifiersvc.predict(X_test)

#Accuracy 
svm_accuracy = round(classifiersvc.score(X_train, y_train) * 100, 2)
svm_accuracy
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifiersvc, X = X_train, y = y_train, cv = 10)
svmk = round(accuracies.mean() *100,ndigits=2)
svmk
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifiernb = GaussianNB()
classifiernb.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifiernb.predict(X_test)

#Accuracy 
nb_accuracy = round(classifiernb.score(X_train, y_train) * 100, 2)
nb_accuracy
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifiernb, X = X_train, y = y_train, cv = 10)
nbk = round(accuracies.mean() *100,ndigits=2)
nbk
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifierdt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifierdt.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifierdt.predict(X_test)


#Accuracy 
dt_accuracy = round(classifierdt.score(X_train, y_train) * 100, 2)
dt_accuracy
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifierdt, X = X_train, y = y_train, cv = 10)
dtk = round(accuracies.mean() *100,ndigits=2)
dtk
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifierrf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifierrf.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifierrf.predict(X_test)

#Accuracy 
rf_accuracy = round(classifierrf.score(X_train, y_train) * 100, 2)
rf_accuracy
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifierrf, X = X_train, y = y_train, cv = 10)
rfk = round(accuracies.mean() *100,ndigits=2)
rfk
bmodels = pd.DataFrame({
    'Model': ['Support Vector Machines', 'K-Nearest Neighbour', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Decision Tree'],
    'Score': [svm_accuracy, knn_accuracy, lr_accuracy, 
              rf_accuracy, nb_accuracy, dt_accuracy],
    'K-Fold' :[lrk, knnk, svmk, nbk, dtk, rfk]})
models.sort_values(by='Score', ascending=False)