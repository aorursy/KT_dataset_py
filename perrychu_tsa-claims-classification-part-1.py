import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
#Read Kaggle file
df = pd.read_csv("../input/tsa_claims.csv",low_memory=False)

#Format columns nicely for dataframe index
df.columns = [s.strip().replace(" ","_") for s in df.columns]

#Rename date columns
df["Date_Received_String"] = df.Date_Received
df["Incident_Date_String"] = df.Incident_Date
df.drop(["Date_Received","Incident_Date"], axis=1, inplace=True)

print("Rows:", len(df))
# Check distribution of nulls per row
temp = df.isnull().sum(axis=1).value_counts().sort_index()
print ("Nulls    Rows     Cum. Rows")
for i in range(len(temp)):
    print ("{:2d}: {:10d} {:10d}".format(temp.index[i], temp[i], temp[i:].sum()))
# Check distribution of nulls per column
df.isnull().sum().sort_values(ascending=False)
#Drop rows with too many nulls
df.dropna(thresh=6, inplace=True)

#Fill NA for categorical columns
fill_columns = ["Airline_Name","Airport_Name","Airport_Code","Claim_Type","Claim_Site","Item"]
df[fill_columns] = df[fill_columns].fillna("-")

#Set NA Claim Amount to 0. Zeros are dropped later in the code.
df["Claim_Amount"] = df.Claim_Amount.fillna("$0.00")

#Dropping these nulls later on:
#  Incident Date / Date Received
#  Status

print(len(df))
df.Status.str.split(";").map(lambda x: "Null" if type(x)==float else x[0]).value_counts()
valid_targets = ['Denied','Approved','Deny','Settled','Approve in Full', 'Settle']

df = df[df.Status.isin(valid_targets)]
df.Status.replace("Approve in Full","Approved",inplace=True)
df.Status.replace("Deny","Denied",inplace=True)
df.Status.replace("Settle","Settled",inplace=True)

print(df.Status.value_counts())
print(len(df))
#Drop nulls
df.dropna(subset=["Date_Received_String"], inplace=True)

#Format datetime
df["Date_Received"] = pd.to_datetime(df.Date_Received_String,format="%d-%b-%y")

#Check year range
df = df[df.Date_Received.dt.year.isin(range(2002,2014+1))]

print(df.Date_Received.dt.year.value_counts().sort_index())
month_dict = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}

def format_dates(regex, date_string):
    '''
    Formats the date string from 2014 entries to be consistent with the rest of the doc 
    Inputs: 
        regex - compiled re with three groups corresponding to {day}/{month (abbrev.)}/{Year}
        date_string - string to be formatted matching the regex
    Outputs: 
        If regex match, return formatted string of form {Month}/{Day}/{Year}; else return original string
    '''
    m = regex.match(date_string)
    if(m):
        day, month, year = m.group(1,2,3)
        return "{}/{}/{}".format(month_dict[month],day,"20"+year)
    else:
        return date_string
        
#Drop nulls
df.dropna(subset=["Incident_Date_String"], inplace=True)

#Error correction for one value in Kaggle data set (looked up in original TSA data)
df.Incident_Date_String.replace("6/30/10","06/30/2010 16:30",inplace=True)

#String formatting for consistency
df["Incident_Date_String"] = df.Incident_Date_String.str.replace("-","/")
df["Incident_Date_String"] = df.Incident_Date_String.str.lower()

#Splitting up time (if exists otherwise will be date) and date components
df["Incident_Time"] = df.Incident_Date_String.str.split(" ").map(lambda x: x[-1])
df["Incident_Date"] = df.Incident_Date_String.str.split(" ").map(lambda x: x[0])

#Could not find a reasonable translation for these entries... most look like "02##"
regex = re.compile(r"/[a-z]{3}/[0-9]{4}")
df = df[df.Incident_Date.map(lambda x: not bool(regex.search(x)))].sort_values(["Date_Received"])

#These are entries received in 2014. Formatting is different from other years but internally consistent.
regex = re.compile(r"(\d*)/([a-z]{3})/(1[1-4])$")
df["Incident_Date"] = df.Incident_Date.map(lambda x: format_dates(regex,x) )
#df[df.Incident_Date.map(lambda x: bool(regex.search(x)))].sort_values(["Date_Received"])

#Format datetime, check year range, create year and month
df["Incident_Date"] = pd.to_datetime(df.Incident_Date,format="%m/%d/%Y")
df = df[df.Incident_Date.dt.year.isin(range(2002,2014+1))]

print(df.Incident_Date.dt.year.value_counts().sort_index())
print(len(df))
#Check multiple Airport Names assigned to one Airport Code
temp = df.groupby("Airport_Code").Airport_Name.nunique().sort_values(ascending=False)
print(df[df.Airport_Code.isin(temp[temp>1].index)].groupby("Airport_Code").Airport_Name.unique().head())
print("\n---\n")

#Duplicates are from excess spaces
df["Airport_Code"] = df.Airport_Code.str.strip()
df["Airport_Name"] = df.Airport_Name.str.strip()

#Check multiple Airport Names assigned to one Airport Code
temp = df.groupby("Airport_Code").Airport_Name.nunique().sort_values(ascending=False)
print(df[df.Airport_Code.isin(temp[temp>1].index)].groupby("Airport_Code").Airport_Name.unique().head())

#Look at tail distribution of claims by airport
temp = df.Airport_Code.value_counts()
print("Total: {} airports, {} complaints".format(temp.count(),temp.sum()))
for num in range(1000,1,-100):
    print("Under {}: {} airports, {} complaints".format(num, temp[temp<num].count(),temp[temp<num].sum()))

level = 200
#plot distribution below level
#temp[temp<level].count(), temp[temp<level].sum()
#temp[temp<level].plot.bar()

#Set airport and code to "Other" under level
def set_other(row, keep_items):
    if row.Airport_Code in keep_items:
        row["Airport_Code_Group"] = row.Airport_Code
        row["Airport_Name_Group"] = row.Airport_Name
    else:
        row["Airport_Code_Group"] = 'Other'
        row["Airport_Name_Group"] = 'Other'
    return row

keep_set = set(temp[temp>=level].index)
df = df.apply(lambda x: set_other(x,keep_set),axis=1)
df["Airline_Name"] = df.Airline_Name.str.strip().str.replace(" ","")
df.Airline_Name.replace("AmericanEagle","AmericanAirlines",inplace=True)
df.Airline_Name.replace("AmericanWest","AmericaWest",inplace=True)
df.Airline_Name.replace("AirTranAirlines(donotuse)","AirTranAirlines",inplace=True)
df.Airline_Name.replace("AeroflotRussianInternational","AeroFlot",inplace=True)
df.Airline_Name.replace("ContinentalExpressInc","ContinentalAirlines",inplace=True)
df.Airline_Name.replace("Delta(Song)","DeltaAirLines",inplace=True)
df.Airline_Name.replace("FrontierAviationInc","FrontierAirlines",inplace=True)
df.Airline_Name.replace("NorthwestInternationalAirwaysLtd","NorthwestAirlines",inplace=True)
df.Airline_Name.replace("SkywestAirlinesAustralia","SkywestAirlinesIncUSA",inplace=True)

df.Airline_Name.value_counts().head(10)
print(len(df))
print(df.Claim_Type.value_counts())
print(df.Claim_Site.value_counts())
#Isolating broadest item categories
#Items column is a text list of all item categories. Sub categories are inconsistent across years.
df_item = df.Item.str.split("-").map(lambda x: "" if type(x) == float else x[0])
df_item = df_item.str.split(r" \(").map(lambda x: x[0])
df_item = df_item.str.split(r" &").map(lambda x: x[0])
df_item = df_item.str.split(r"; ").map(lambda x: x[0])
df_item = df_item.str.strip()

categories = df_item.value_counts()

#categories[[not bool(re.compile(";").search(x)) for x in categories.index]][0:]

categories[categories > 100]
df["Claim_Amount"] = df.Claim_Amount.str.strip()
df["Claim_Amount"] = df.Claim_Amount.str.replace(";","").str.replace("$","").str.replace("-","0")
df["Claim_Value"] = df.Claim_Amount.astype(float)

df_copy = df.copy()

print(df.Claim_Value.describe())
print(df.Status.value_counts())
print(len(df))

sns.distplot(df.Claim_Value[(df.Claim_Value>0)&(df.Claim_Value<500)])

df.Status[(df.Claim_Value>0)&(df.Claim_Value<1000)].value_counts()
bins = [round(10**x) for x in (list(np.arange(0,4.1,.4))+[10])]

bottom = -1

data = []

for x,top in enumerate(bins):
    counts = df.Status[(df.Claim_Value>bottom)&(df.Claim_Value<=top)].value_counts()
    for i in range(len(counts)):
        data.append({"bin":(str(x)+":"+str(top)),"label":counts.index[i],"count":counts[i]})
    bottom = top

counts_df = pd.DataFrame(data)

sns.factorplot(x="bin",y="count",hue="label",data=counts_df,kind="bar",size=10)
df = df[df.Claim_Value != 0]

print(df.Claim_Value.describe())
print(df.Status.value_counts())
print(len(df))
df["Close_Amount"] = df.Close_Amount.str.strip()
df["Close_Amount"] = df.Close_Amount.str.replace(";","").str.replace("$","")
df["Close_Value"] = df.Close_Amount.astype(float)
df.Close_Value.describe()
plot_df = df[(df.Claim_Value < 200000) & (df.Close_Value <= 500000)]

plt.scatter(plot_df.Claim_Value,plot_df.Close_Value,alpha=.2)
plt.title("Combined")
plt.xlabel("Claim value")
plt.ylabel("Close value")
plt.show()

fig,ax = plt.subplots(1,3)
fig.set_size_inches(16,4)

for i,s in enumerate(plot_df.Status.unique()):
    ax[i].scatter(plot_df[plot_df.Status==s].Claim_Value,plot_df[plot_df.Status==s].Close_Value,alpha=.2)
    ax[i].set_title(s)

output_df = df.drop(["Close_Amount", "Claim_Amount", "Disposition",
                     "Date_Received_String","Incident_Date_String","Incident_Time",
                     "Airport_Code","Airport_Name"],axis=1)

output_df.to_csv("tsa_claims_clean.csv",index=False)

output_df.head(5)