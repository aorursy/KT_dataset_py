import pandas as pd
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
traindf = pd.read_csv("/kaggle/input/tmdb-box-office-prediction/train.csv")
testdf = pd.read_csv("/kaggle/input/tmdb-box-office-prediction/test.csv")
print(traindf.shape)
print(testdf.shape)
traindf.head(5)
# Null_Values = traindf.isnull()
# for column in Null_Values.columns.values.tolist():
#     print(Null_Values[column].value_counts())
#     print("")
print("Train Data",traindf.isnull().sum())
print("\n\n\n\nTest Data",testdf.isnull().sum())
   
    
print("Train Data",traindf.info())
print("\n\n\n\n")
print("Test Data",testdf.info())
traindf.describe(include = 'all')
testdf.describe(include = 'all')
BinsBudget = numpy.linspace(min(traindf["budget"]), max(traindf["budget"]),4)
BinsRun = numpy.linspace(min(traindf["runtime"]), max(traindf["runtime"]),4)

BinsBudgetName = ["Low Budget", "Medium Budget", "High Budget"] 
BinsRunName = ["Short ", "Medium", "Long "]

traindf["BinBudget"] = pd.cut(traindf["budget"],BinsBudget,labels = BinsBudgetName,include_lowest = True)
traindf["BinRunTime"] = pd.cut(traindf["runtime"],BinsRun,labels = BinsRunName,include_lowest = True)
traindf

plt.figure(figsize = (15,8))
sns.countplot(traindf["runtime"])
plt.xlabel("Runtime")
plt.ylabel("Count")
plt.title("Runtime Variations of Box Office")
plt.figure(figsize = (15,8))
sns.countplot(traindf["BinRunTime"].sort_values())
plt.xlabel("RunTime")
plt.ylabel("Count")
plt.title("RunTime Variations of Box Office")
sns.jointplot(x="budget", y="revenue", data=traindf, height=11, ratio=6, color="r")
plt.xlabel("RunTime")
plt.ylabel("Count")
plt.title("RunTime Variations of Box Office")
plt.figure(figsize = (15,8))
plt.hist(traindf["budget"])
plt.xlabel("Budget")
plt.ylabel("Count")
plt.title("Budget Variations of Box Office")
plt.figure(figsize = (15,8))
plt.hist(traindf["BinBudget"])
plt.xlabel("Budget")
plt.ylabel("Count")
plt.title("Budget Variations of Box Office")
plt.figure(figsize = (25,8))
# plt.scatter("budget","revenue",data=traindf)
sns.jointplot(x="budget", y="revenue", data=traindf, height=11, ratio=6, color="g")
plt.xlabel("Budget")
plt.ylabel("Count")
plt.title("Budget Variations of Box Office")
plt.figure(figsize = (15,8))
plt.hist(traindf["popularity"])
plt.xlabel("Runtime")
plt.ylabel("Count")
plt.title("Runtime Variations of Box Office")
plt.show()
plt.figure(figsize = (15,8))
sns.countplot(traindf["popularity"])
plt.xlabel("Runtime")
plt.ylabel("Count")
plt.title("Runtime Variations of Box Office")
plt.show()
plt.figure(figsize = (15,8))
# plt.scatter(traindf["popularity"],traindf["revenue"])
sns.jointplot(traindf["popularity"],traindf["revenue"],height = 12,ratio =6,color = "b")

plt.xlabel("Runtime")
plt.ylabel("Count")
plt.title("Runtime Variations of Box Office")
plt.show()
sns.pairplot(traindf[["revenue","popularity","runtime","budget"]])
plt.figure(figsize = (15,8))
sns.countplot(traindf["original_language"])
plt.xlabel("original Language")
plt.ylabel("Count")
plt.title("Original Language of Box Office")
plt.show()
test = traindf[["BinBudget","BinRunTime","revenue"]]
Group = test.groupby(["BinBudget","BinRunTime"],as_index=False).mean()
Pivot = Group.pivot("BinBudget","BinRunTime")

display(Group)
display(Pivot)

plt.figure(figsize = (15,8))
sns.heatmap(Pivot,annot=True, cmap="RdBu")
plt.title("Heatmap of Binned Budget & Runtime with Price")
plt.figure(figsize = (25,10))

plt.scatter(traindf["budget"],traindf["runtime"],c=traindf["revenue"],cmap="RdGy")
plt.colorbar()
plt.title("Heatmap of Budget & Runtime with Price with scatter")
plt.xlabel("Runtime")
plt.ylabel("Budget")
plt.figure(figsize = (25,12))
sns.heatmap(traindf.corr(),annot = True)
traindf[['release_month','release_day','release_year']]=traindf['release_date'].str.split('/',expand=True).replace(numpy.nan, -1).astype(int)
traindf.head(2)
plt.figure(figsize = (25,12))
sns.heatmap(traindf.corr(),annot = True)
plt.figure(figsize = (15,8))
sns.countplot(traindf["release_month"])
plt.xlabel("Release Month")
plt.ylabel("Count")
plt.title("Monthly Release Variation")
plt.show()
plt.figure(figsize = (15,8))

test2 = traindf[["release_month","budget"]]
group2 = test2.groupby(["release_month"]).mean()
plt.plot(group2)

plt.xlabel("Release Month")
plt.ylabel("Budget")
plt.title("Average Monthly Budget Variation")
plt.show()
plt.figure(figsize = (15,8))

test3 = traindf[["release_month","revenue"]]
group3 = test3.groupby(["release_month"]).mean()
plt.plot(group3)

plt.xlabel("Release Month")
plt.ylabel("Revnue")
plt.title("Average Monthly Revenue Variation")
plt.show()
plt.figure(figsize = (18,8))
sns.countplot(traindf["release_day"])
plt.xlabel("Release Month")
plt.ylabel("Count")
plt.title("Day Release Variation")
plt.show()
plt.figure(figsize = (15,8))

test4 = traindf[["release_day","budget"]]
group4 = test4.groupby(["release_day"]).mean()
plt.plot(group4)

plt.xlabel("Release Day")
plt.ylabel("Budget")
plt.title("Average Daily Budget Variation")
plt.show()
plt.figure(figsize = (15,8))

testdaily = traindf[["release_day","revenue"]]
groupdaily = testdaily.groupby(["release_day"]).mean()
plt.plot(groupdaily)

plt.xlabel("Release Day")
plt.ylabel("Revnue")
plt.title("Average Daily Revenue Variation")
plt.show()
plt.figure(figsize = (30,8))
sns.countplot(traindf["release_year"])
plt.xlabel("Release Month")
plt.ylabel("Count")
plt.title("Yearly Release Variation")
plt.show()
plt.figure(figsize = (15,8))

traindf['YearlyBudget'] = traindf[["release_month","budget"]].groupby(["release_month"]).mean()
plt.plot(traindf['YearlyBudget'])

plt.xlabel("Release Year")
plt.ylabel("Revnue")
plt.title("Average Yearly Revenue Variation")
plt.show()
plt.figure(figsize = (15,8))

traindf['YearlyRevnue'] = traindf[["release_month","revenue"]].groupby(["release_month"]).mean()
plt.plot(traindf['YearlyRevnue'])

plt.xlabel("Release Day")
plt.ylabel("Revnue")
plt.title("Average Daily Revenue Variation")
plt.show()
sns.pairplot(traindf[["budget","revenue","release_year","release_month","release_day"]])
traindf.loc[pd.isnull(traindf["homepage"]) ,"Is There Homepage"] = 0
traindf["Is There Homepage"].replace(numpy.nan,"1",inplace=True)    # So int this NaN will be for the title that have home page

traindf.loc[pd.isnull(traindf["tagline"]) ,"Is There Tagline"] = 0
traindf["Is There Tagline"].replace(numpy.nan,"1",inplace=True)    # So int this NaN will be for the title that have Tagline

traindf['English Movie'] = 0 
traindf.loc[ traindf['original_language'] == "en" ,"English Movie"] = 1

traindf
plt.figure(figsize = (15,7))
sns.countplot(traindf["Is There Homepage"])
# plt.figure(figsize = (28,17))
sns.catplot("Is There Homepage",y="revenue",data = traindf)
plt.figure(figsize = (28,17))
sns.catplot("Is There Tagline",y="revenue",data = traindf)


from sklearn.linear_model import LinearRegression 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


TrainingFeatures = traindf[["budget","popularity","runtime","Is There Homepage","Is There Tagline","English Movie"]]
y = traindf["revenue"]

xTrain,xTest,yTrain,yTest = train_test_split(TrainingFeatures,y,test_size = 0.10,random_state = 0)
TrainingFeatures.isnull().sum()
traindf["runtime"].replace((numpy.nan,),traindf["runtime"].mean(),inplace=True) 

# Replacing the Null Values by mean of the column

TrainingFeatures.isnull().sum()
lr = LinearRegression()
lr.fit(xTrain,yTrain)
print("Rsquared Score for Linear Regression Model = ", lr.score(xTest,yTest))
RsqTest = []
order = numpy.arange(1,6,1)

for n in order:
    pr = PolynomialFeatures(degree = n )
    Scale = StandardScaler()
    xTrainTrans = pr.fit_transform(xTrain) 
    xTestTrans = pr.fit_transform(xTest)
    
    
    lr.fit(xTrainTrans,yTrain)
    RsqTest.append(lr.score(xTestTrans,yTest))
        
display(RsqTest) 

# plt.plot(RsqTest)
# plt.ylim(0.6,0.7)
input = [["Scale",StandardScaler()],["Poly",PolynomialFeatures(degree=3)],["lr",LinearRegression()]]
Pipe = Pipeline(input)
Pipe.fit(xTrain,yTrain)
Pipe.score(xTest,yTest)
Pipe.score(xTrain,yTrain)
Pipe.score(TrainingFeatures,y)
Pipe.fit(TrainingFeatures,y)
# Pipe.score(xTest,yTest)
# Pipe.score(xTrain,yTrain)
Pipe.score(TrainingFeatures,y)
RsqTrain2 = []
RsqTest2 = []

ALFAVAL = [0.000000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000,1000000,10000000]

for m in ALFAVAL:
    RidgeModel = Ridge(alpha=m)
    
    RidgeModel.fit(xTrain,yTrain)
    
    RsqTest2.append(RidgeModel.score(xTest,yTest))
    RsqTrain2.append(RidgeModel.score(xTrain,yTrain))

display(RsqTest2) 
# display(RsqTrain2)
# plt.plot(RsqTest2)
ScoresCVS = cross_val_score(Pipe,TrainingFeatures,y,cv = 17) 
numpy.mean(ScoresCVS)
testdf.head(2)
testdf.loc[pd.isnull(testdf["homepage"]) ,"Is There Homepage"] = 0
testdf["Is There Homepage"].replace(numpy.nan,"1",inplace=True)    # So int this NaN will be for the title that have home page

testdf.loc[pd.isnull(testdf["tagline"]) ,"Is There Tagline"] = 0
testdf["Is There Tagline"].replace(numpy.nan,"1",inplace=True)    # So int this NaN will be for the title that have Tagline

testdf['English Movie'] = 0 
testdf.loc[ testdf['original_language'] == "en" ,"English Movie"] = 1
testdf.head(2)
testdf["budget"].replace(0,testdf["budget"].mean(),inplace=True)
TestingFeatures.isnull().sum()
testdf["runtime"].replace(numpy.nan, testdf["runtime"].mean(),inplace = True)
TestingFeatures = testdf[["budget","popularity","runtime","Is There Homepage","Is There Tagline","English Movie"]]

Pipe.fit(TrainingFeatures,y)

Predicted_Revenue = Pipe.predict(TestingFeatures)
Predicted_Revenue = numpy.array(Predicted_Revenue)



SubFile = pd.read_csv("../input/tmdb-box-office-prediction/sample_submission.csv")
SubFile["revenue"] = Predicted_Revenue
SubFile
SubFile.to_csv("My Final Sub.csv",index=False)


