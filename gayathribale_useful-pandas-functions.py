from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# importing librrary
import pandas as pd  # pd is the alias name,we can use any name other than pd as well
series =pd.Series([3, -10, 7, 4])  
series  # By default we have continous numbers,index starts from Zero
type(series)   #Which return type of series
#we can change the index with names as well by using index attribute
series =pd.Series([3, -10, 7, 4],index=['a', 'b', 'c', 'd']) 
series
#Creating a dictionary where each key will be a DataFrame column

data = {
'person': ['John Smith', 'David jones', 'Juan Carlos','Mike Jones'],
'first name': ['John', 'David', 'Juan', 'Mike'],
'Last Name': ['Smith', 'jones', 'Carlos','Jones'],
'email id': ["JohnSmith@gmail.com","Davidjones@gmail.com","JuanCarlos@gmail.com","MikeJones@gmail.com"]  
        }
data["person"],data["first name"],data['Last Name']
#Converting a dictionary to a data frame 
import pandas as pd
df=pd.DataFrame(data)
df   #by default we have a index starting from Zero
df.index    # Which returns row index 
df.columns  # Which returns Columns in a dataframe
df.shape    # shape is a attribute which returns dimentions 
type(df)    # returns type of df
df.dtypes   # returns data type of a columns based up on the data it holds
            # (eg: person column which has all string therefore it returns object data type )
df.info()   # info() function gives the details of data frame 
# fetching data from a data frame using loc 
#fetching details of John Smith from a df
df.loc[0]                            #forst row details 
df.loc[0:1]                          # first and second row details 
df.loc[1:2]                          # Subset from the data frame 
df.loc[:,"person"] 
df.loc[:,"person":"Last Name"]
# fetching data from a data frame using iloc
df.iloc[0:1] # 0 is inclusive and 1 is exclusive
df.iloc[0:1,0:3]   # i stands for indexing 
df.iloc[0:1,0] # this will return error as iloc doesn't support label concept 

#index is always a unique so replacind index with email id by using df.set_index function
df.set_index("email id")   # whcih is temparary 
df
#set_index
df.set_index("email id",inplace=True)  #  used inplace to change permanently
df
df   #index replaced by email id using set_index and making it permanent fix by using inplace=True
# Readinng a CSV file and used index_col to set the index as respondent 

df_csv=pd.read_csv("../input/stack-overflow-2018-developer-survey/survey_results_public.csv", index_col="Respondent") 
df_csv.head() # returns first five records from a CSV file
df_csv.shape # returns total number of rows and coulmn 
#set_option
pd.set_option('display.max_columns',85)
df_scheme=pd.read_csv("../input/stack-overflow-2018-developer-survey/survey_results_schema.csv" ,encoding='utf-8', index_col="Column")
df_scheme.head()
df_scheme.loc["Respondent"]   # which give the detailed discription of Respondent with truncate
pd.set_option('display.max_columns',85)

df_scheme.loc["Respondent","QuestionText"]   
pd.set_option('display.max_rows',85)
df_scheme.sort_index(ascending=False)
df
filt=(df["Last Name"]=="Jones")   # filtering data from a data Frame 
df[filt]
data = {
'person': ['John Smith', 'David jones', 'Juan Carlos','Mike Jones'],
'first name': ['John', 'David', 'Juan', 'Mike'],
'Last Name': ['Smith', 'jones', 'Carlos','Jones'],
'email id': ["JohnSmith@gmail.com","Davidjones@gmail.com","JuanCarlos@gmail.com","MikeJones@gmail.com"]  
        }
import pandas as pd
df1=pd.DataFrame(data)
df1
df1.columns
df1.columns=["Person","First","Last","Email"]
df1
df1.columns=[x.upper() for x in df1.columns ]
df1
df1.columns=["PERSON_1","FIRST_Name","LAST_Name","EMAIL_ID"]
df1
df1.columns=df1.columns.str.replace('_'," ")
df1
df1.columns=df1.columns.str.replace(' ',"_")
df1
# Changing only a specific column using dictionary 
df1.columns=[x.lower() for x in df1.columns ]
df1
df1.rename(columns={"person_1":"person","first_name":"first","last_name":"last"}, inplace=True)
df1
#Updating a specific column using list
df1.loc[2]=["Mike Jones","Mike", "Carlos","MikeJones@gmail.com"]
df1
df1.at[2,["person","first","email_id"]]=["Juan Carlos","Juan","JuanCarlos@gmail.com"]
df1
df1.loc[2,["person","first","email_id"]]=["Juan Carlos","Juan","JuanCarlos@gmail.com"]
df1
filt = df1["email_id"]=="MikeJones@gmail.com"
df1[filt]["last"]="Smith"    # which is not possible to update the column 
df1
df1.loc[filt,"last"]="smith"
df1
df1.loc[filt,"last"]="Jones"
df1
df1["email_id"]=df1["email_id"].str.lower()   # Changing lower case of a sepcific column using string method 

df1
#apply is used for call a function
#map
#apply map
#Replace

df1["person"].apply(len)    
def update_email(email_id):
    return email_id.upper()
    
df1["email_id"]=df1["email_id"].apply(update_email)
df1["email_id"]=[x.lower() for x in df1["email_id"]]
df1
df1["first"]=df1["first"].apply(lambda x : x.lower())
df1
df1["last"]=df1["first"].apply(lambda x : x.lower()) 
df1
df1.apply(len)
df1.apply(len,axis="columns")  
len(df1["person"])
df1.apply(min)
df1.apply(max)
df1.apply(pd.Series.min)
df1.apply(pd.Series.max)
#Lambda works on seriesobject 
df1.apply(lambda x:x.min())

df1.applymap(len)
df1.applymap(str.lower)
df1["person"].str.upper()
df1["person"].apply(str.upper)
df1
df1["first"].map({"john":"jon","david":"doe"})
df1["first"].replace({"john":"jon","david":"doe"})
df1["first"]=df1["first"].replace({"john":"jon","david":"doe"})
df1
df_csv=pd.read_csv("../input/stack-overflow-2018-developer-survey/survey_results_public.csv", index_col="Respondent") 
df_csv.head() # returns first five records from a CSV file
df_csv.shape # returns total number of rows and coulmn 
#set_option
pd.set_option('display.max_columns',85)
df_scheme=pd.read_csv("../input/stack-overflow-2018-developer-survey/survey_results_schema.csv", index_col="Column")
df_scheme.head()
pd.set_option('display.max_columns',85)
df_scheme.loc['Respondent',"QuestionText"]
df_csv.head()
#Only a particular column
df_csv.rename(columns={"ConvertedComp":"SalaryUSD"}, inplace=True)
df_csv.columns
df_csv["Hobby"].map({"Yes":"True","No":"False"})
df_csv["Hobby"]=df_csv["Hobby"].map({"Yes":"True","No":"False"})
df_csv.head()
df1
df1
df1["Full name"]=df1["first"]+" "+df1["last"]
df1
df1.drop(columns=["last"],inplace=True)
df1
df2=["john","david",'juan',"mike"]
df2=pd.DataFrame(df2)
df2.rename(columns={0:"Last"},inplace=True)
df2["Last"]
df1["last"]=df2["Last"]
df1
df1.drop(columns=["last","first"], inplace=True)
df1["person"].str.split(" ", expand=True)
df1[["first_name","Last_Name"]]=df1["person"].str.split(" ", expand=True)
df1
df1.drop(columns="Full name", inplace=True)
df1
df1
df1=df1.append({"first_name":"Tony"},ignore_index=True)  

df1
df1.drop(index=4, inplace=True)
df1
df1.drop(index=df1[df1["Last_Name"]=="Jones"].index)
        
df1
df1.sort_values(by="first_name",  ascending=False)    

df1.sort_values(by="first_name") 
df1.sort_values(by=["first_name" ,"Last_Name"] , ascending=[False,True], inplace=True) 
df1
df1.sort_index(inplace=True)
df1
df_csv=pd.read_csv("../input/stack-overflow-2018-developer-survey/survey_results_public.csv", index_col="Respondent") 
df_csv.head()
pd.set_option("display.max_columns",85)
df_csv.head(2)
df_scheme=pd.read_csv("../input/stack-overflow-annual-developer-survey-2019/survey_results_public.csv", index_col="Column")
df_scheme.head()
df_csv.sort_values(by=["Country","CompanySize"],ascending=[True,False],inplace=True)

df_csv[["Country","CompanySize"]].head(10)
df_csv["AssessJob1"].nlargest(10)
df_csv.nsmallest(10,"AssessJob1")
df_csv.nlargest(10,"AssessJob1")
df_csv.head(2)
df_csv["AssessJob1"].median()
df_csv.median()
df_csv.describe()
df_csv.columns
df_csv["Hobby"].value_counts() 
df_csv["SocialMedia"]
df_scheme.loc["SocialMedia"]
df_csv["SocialMedia"].value_counts()
df_csv["SocialMedia"].value_counts(normalize=True)
df_csv["Country"].value_counts()
Country_GrpBy=df_csv.groupby(["Country"])
Country_GrpBy.get_group("United States")
Country_GrpBy["SocialMedia"].value_counts()
Country_GrpBy["SocialMedia"].value_counts(normalize=True).loc["China"]
Country_GrpBy["ConvertedComp"].median().loc["Germany"]
Country_GrpBy["ConvertedComp"].median()
Country_GrpBy["ConvertedComp"].mean().loc["Germany"]
Country_GrpBy["ConvertedComp"].agg(["median","mean"])   # by using agg method 
Country_GrpBy["ConvertedComp"].agg(["median","mean"]).loc["Canada"]
filt=df_csv["Country"]=="India"
df_csv.loc[filt]["LanguageWorkedWith"].str.contains("Python").sum()
Country_GrpBy["LanguageWorkedWith"].apply(lambda x: x.str.contains("Python").sum()).loc["India"]
Country_GrpBy["LanguageWorkedWith"].apply(lambda x: x.str.contains("Python")).value_counts(normalize=True)
Country_Responded=df_csv["Country"].value_counts()
Country_Responded
Country_use_Python=Country_GrpBy["LanguageWorkedWith"].apply(lambda x: x.str.contains("Python").sum())
Country_use_Python
python_df=pd.concat([Country_Responded,Country_use_Python,], axis="columns", sort=False)
python_df
python_df.rename(columns={"Country":"Number_of_respondent","LanguageWorkedWith":"PersonKnowsPython"}, inplace=True)
python_df
python_df["PerKnowPython"]=(python_df['PersonKnowsPython']/python_df['Number_of_respondent']) * 100
python_df
python_df.sort_values("PerKnowPython", ascending=False, inplace=True)
python_df
python_df.loc["Japan"]
df1=df1.append({"first_name":"Tom"}, ignore_index=True)
df1
df1
df1.dropna()
df1
df1.dropna(axis='index',how="any")    # by default it has axis= index and how = "any"
df1.dropna(axis='index',how="all", subset=["first_name"])
df1.dropna(axis='index',how="any", subset=["first_name"])
df1.dropna(axis='index',how="any", subset=["email_id"])
df1.dropna(axis='index',how="any", subset=["first_name","email_id"])
df1
import numpy as np
import pandas as pd
data1 = {
'person': ['Mising', 'NA', 'NAN','Mike Jones'],
'first': ['John', 'David', 'Juan', 'Mike'],
'email_id': ["JohnSmith@gmail.com","Davidjones@gmail.com","Mising","NA"],
 'Full name':["Missing",np.nan,None,"NA"]   
        }
df2=pd.DataFrame(data1)
df2
data = {
'person': ['John Smith', 'David jones', 'Juan Carlos','Mike Jones'],
'first name': ['John', 'David', 'Juan', 'Mike'],
'Last Name': ['Smith', 'jones', 'Carlos','Jones'],
'email id': ["JohnSmith@gmail.com","Davidjones@gmail.com","JuanCarlos@gmail.com","MikeJones@gmail.com"]  
        }
df1=pd.DataFrame(data)
df1
df2
df1.rename(columns={"first name":"first","email id":"email_id","Last Name":"Full name"}, inplace=True)
df1=df1.append(df2)
df1
df1
df1.dropna()
df1 # bydefault axis="index"  and how ="any" 
df1.replace("NA",np.nan,inplace=True)
df1
df1.replace(["Mising","Mising","None"],np.nan,inplace=True)
df1
df1.dropna()
df1.isna()
df1.fillna("MISSING")
df1.dtypes
type(np.nan)
df_csv.head(2)
df_csv["YearsCode"].unique()
df_csv.replace("Less than 1 year",0,inplace=True)
df_csv.replace('More than 50 years',51,inplace=True)
df_csv["YearsCode"]=df_csv["YearsCode"].astype(float)
df_csv["YearsCode"].dtypes
df_csv["YearsCode"].mean(),df_csv["YearsCode"].median()
df_csv["YearsCode"].mode()
import pandas as pd 
d_parse= lambda x: pd.datetime.strptime(x, "%Y-%m-%d %I-%p")
df_TimeStamp=pd.read_csv("../input/time-series-data-set/ETH_1H.csv",parse_dates=["Date"],date_parser=d_parse)
df_TimeStamp
df_TimeStamp.head()
df_TimeStamp["DayOfWeek"]=df_TimeStamp["Date"].dt.day_name()
df_TimeStamp.head()
df_TimeStamp["Date"].min()   ,    df_TimeStamp["Date"].max()
delta = df_TimeStamp["Date"].max()-df_TimeStamp["Date"].min() 
delta
filt=(df_TimeStamp["Date"] >=pd.to_datetime("2019-01-01"))& (df_TimeStamp["Date"] <pd.to_datetime("2020-01-01"))
df_TimeStamp.loc[filt]
df_TimeStamp.set_index("Date", inplace=True)
df_TimeStamp['2020']
df_TimeStamp['2020-01' : '2020-02']["Close"]
df_TimeStamp["2020-01-01"]["High"].max()
highs=df_TimeStamp["High"].resample("D").max()
highs["2020-01-01"]
%matplotlib inline
highs.plot()
df_TimeStamp.resample("W").mean()
df_TimeStamp[["Close","High","Low","Volume"]].resample("w").agg({"Close":"mean","high":"max","Low":"min","Volume":"sum"})
df_TimeStamp
df_TimeStamp.resample("w").agg(["mean","max","min","sum"]).loc[:,["Close","high","Low","Volume"]]

