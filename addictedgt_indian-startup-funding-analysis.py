#Import the libraries



#1: Pandas will be used for data maniplation of csv file.

import pandas as pd

#2: Numpy adds support for matrix calculation.

import numpy as np

#3: Matplotlib used for creating plots.

import matplotlib.pyplot as plt

#4: Seaborn is used for data visualization. 

import seaborn as sns

#5: Squarify is used to create square plots.

import squarify

#6: RE is used for regular expressions.

import re 

#Read the data

df = pd.read_csv('/kaggle/input/indian-startup-funding/startup_funding.csv')
#Find the type of data in various columns.

df.info()
#Change the investment type into categorical data

df['InvestmentnType'] = df['InvestmentnType'].astype('category')

#Now display the sample of the data.

df.head(10)
# Checking if null values are present in data

df.isnull().sum()
#Remarks is an unnecessary column. So, we remove it.

df = df.drop(columns = 'Remarks')



#Similarly we do away with Sr No.

df = df.drop(columns = 'Sr No')

#Save a copy of original dataframe.

data = df.copy()



#Rename the column for convenieance.

data.columns = ['Date', 'Startup Name', 'Industry Vertical', 'SubVertical',

       'City  Location', 'Investors Name', 'Investment Type',

       'Amount in USD']



#Drop the rows where amount is zero.

data = data.drop(data[data['Amount in USD'].isnull()].index )



#Drop the rows where amount is Undisclosed

data = data.drop(data[(data['Amount in USD'] == 'Undisclosed') | (data['Amount in USD'] == 'undisclosed')].index )



#Data in amount is using delimiter ',' so we need to remove it and convert the data into float. 

data['Amount in USD'] = data['Amount in USD'].apply(lambda x: x.replace(',',''))



#Some data has amount like 25000+ replace the plus to obtain clean data.

data['Amount in USD'] = data['Amount in USD'].apply(lambda x: x.replace('+',''))



#Drop the rows where amount is not a valid float number.

data = data[data['Amount in USD'].str.replace('.','',1).str.isdigit()]



#Convert the data into float

data['Amount in USD'] = data['Amount in USD'].astype('float')
# Fill the missing values with Others.

data['Industry Vertical'] = data['Industry Vertical'].fillna('Others')

data['SubVertical'] = data['SubVertical'].fillna('Others')

data['Industry Vertical'].value_counts()
# Step 1



# Convert all the values to upper case to avoid confusion

data['Industry Vertical'] = data['Industry Vertical'].apply(lambda x: x.upper())



# Replace & with 'AND'

data['Industry Vertical'] = data['Industry Vertical'].apply(lambda x: x.replace('&','AND'))



# A lot of values have \\XA2 replace them with ''

data['Industry Vertical'] = data['Industry Vertical'].apply(lambda x: re.sub('\\\\[A-Z][A-Z][0-9]','',x))



# Now replace \ with ''

data['Industry Vertical'] = data['Industry Vertical'].apply(lambda x: re.sub('\\\\',' ',x))



# Remove any unnecessary spaces.

data['Industry Vertical'] = data['Industry Vertical'].apply(lambda x: x.replace('  ',''))



# Remove - 

data['Industry Vertical'] = data['Industry Vertical'].apply(lambda x: x.replace('-',''))



# I will be using sets for this purpose. I create a set which avoids repeated values.



def create_unique_Set(series):

    # We get all the values in series

    t_List = series.value_counts().index

    

    # We create an empty set to store unique values.

    t_Set = set()

    

    for i in t_List:

        r = re.compile('^{exp}'.format(exp = i.strip()))

        # We match the expression in i with values in temporary set to see if a similar value is already present.

        temp = list(filter(r.match, t_Set))

        if len(temp)>0:

        #If set has similar values we simply replace it with the longest value.    

            temp = [x for x in temp if len(x) == len(max(temp , key = len))]

            if len(temp[0]) < len(i):

                t_Set.remove(temp[0])

                t_Set.add(i)

        # If a match is not found we add the value in set.        

        elif len(temp) == 0 :

            t_Set.add(i)

    return t_Set    



        

# I am using the longest match function to replace similar values.



def find_longest_match(value, set_values):

    r = re.compile('^{val}'.format(val = value))

    # I match the value with values in our unique list.

    temp_list = list(filter(r.match , set_values))

    # We replace the current value with the longest value.

    temp = [x for x in temp_list if len(x) == len(max(temp_list, key = len))]

    # We return the longest value.

    if len(temp) > 0:

        temp = temp[0]

        return temp

    else:

        return value

# Create the set of unique values for data['Industry Vertical'] and eliminate duplicate values. 

set_iv = create_unique_Set(data['Industry Vertical'])      

data['Industry Vertical'] = data['Industry Vertical'].apply(find_longest_match , set_values = set_iv)  



data['Industry Vertical'].value_counts()
# Step 1



# Convert all the values to upper case to avoid confusion

data['SubVertical'] = data['SubVertical'].apply(lambda x: x.upper())



# Replace & with 'AND'

data['SubVertical'] = data['SubVertical'].apply(lambda x: x.replace('&','AND'))



# A lot of values have \\XA2 replace them with ''

data['SubVertical'] = data['SubVertical'].apply(lambda x: re.sub('\\\\[A-Z][A-Z][0-9]','',x))



# Now replace \ with ''

data['SubVertical'] = data['SubVertical'].apply(lambda x: re.sub('\\\\',' ',x))



# Remove any unnecessary spaces.

data['SubVertical'] = data['SubVertical'].apply(lambda x: x.replace('  ',''))



# Remove - 

data['SubVertical'] = data['SubVertical'].apply(lambda x: x.replace('-',''))



# Remove NLOANS COMPARISON PLATFORMNNNN (ADSBYGOOGLE = WINDOW.ADSBYGOOGLE || []).PUSH({});NN with NLOANS COMPARISON PLATFORM

data['SubVertical'] = data['SubVertical'].apply(lambda x: re.sub(r'\([^()]*\)','',x))



# Create the set of unique values for data['Industry Vertical'] and eliminate duplicate values.

set_sv = create_unique_Set(data['SubVertical'])

data['SubVertical'] = data['SubVertical'].apply(find_longest_match , set_values = set_sv)  



data['SubVertical'].value_counts()
# Check for null values

print("Null values are: " , len(data[data['City  Location'].isnull()]))



# Replace the Null values with others.

data['City  Location'] = data['City  Location'].fillna('OTHERS')



# We convert all values to uppercase for convineance.

data['City  Location'] = data['City  Location'].apply(lambda x: x.upper())



# The data has multiple cities seperated by /. So we remove the slash and pick one city. 

data['City  Location'].value_counts()



# We remove the slash and remove any excess space.

data['City  Location'] = data['City  Location'].apply(lambda x: x.split('/')[0].strip())



# We have to clean the data to avoid duplicity. 

data.loc[(data['City  Location'] == 'AHEMADABAD') | (data['City  Location'] == 'AHMEDABAD') , 'City  Location'] = 'AHMEDABAD'

data.loc[(data['City  Location'] == 'BANGALORE') | (data['City  Location'] == 'BENGALURU') , 'City  Location'] = 'BENGALURU'

data.loc[(data['City  Location'] == 'GURGAON') | (data['City  Location'] == 'GURUGRAM') , 'City  Location'] = 'GURUGRAM'



# Display the cleaned data.

data['City  Location']
print(data['Investment Type'].value_counts(dropna = False))



# Function to remove the slash

def remove_Slash(investment):

    temp = investment

    if re.search('/', investment):

        temp = investment.split('/')[1].strip()        

    return temp.upper().strip()    





#Remove the null values with Private Equity as it is the most common values.

data['Investment Type'] = data['Investment Type'].fillna('Private Equity')



# Remove the \\n.

data['Investment Type'] = data['Investment Type'].apply(lambda x: re.sub(r'\\\\n' , ' ' , x))



# Convert all values to uppercase for convineance and remove the /.

data['Investment Type'] = data['Investment Type'].apply(remove_Slash)



# Change the values of seed and seed round to seed funding to avoid duplication.. 

data['Investment Type'][(data['Investment Type'] == 'SEED') | (data['Investment Type'] == 'SEED ROUND') | (data['Investment Type'] == 'SEED FUNDING ROUND')] = 'SEED FUNDING'



# Change angel round, angel and angle funding to angel funding to avoid duplication.

data['Investment Type'][(data['Investment Type'] == 'ANGEL') | (data['Investment Type'] == 'ANGEL ROUND') | (data['Investment Type'] == 'ANGLE FUNDING')] = 'ANGEL FUNDING'



# Change DEBT and DEBT-FUNDING to DEBT FUNDING to avoid duplication.

data['Investment Type'][(data['Investment Type'] == 'DEBT') | (data['Investment Type'] == 'DEBT-FUNDING') ] = 'DEBT FUNDING'



# Change EQUITY and EQUITY BASED FUNDING to EQUITY BASED FUNDING to avoid duplication.

data['Investment Type'][(data['Investment Type'] == 'EQUITY') | (data['Investment Type'] == 'EQUITY BASED FUNDING') ] = 'EQUITY BASED FUNDING'



# Change PRIVATE FUNDING, PRIVATEEQUITY , PRIVATE FUNDING ROUND and PRIVATE to PRIVATE EQUITY to avoid duplication.

data['Investment Type'][(data['Investment Type'] == 'PRIVATE FUNDING') | (data['Investment Type'] == 'PRIVATE EQUITY ROUND') | (data['Investment Type'] == 'PRIVATE') | (data['Investment Type'] == 'PRIVATEEQUITY') | (data['Investment Type'] == 'PRIVATE FUNDING ROUND')] = 'PRIVATE EQUITY'
print('Cleaned data.')

print(data['Investment Type'].value_counts(dropna = False))
# First replace // with / and . with /.

data['Date'] = data['Date'].apply(lambda x: x.replace('//' , '/' ))

data['Date'] = data['Date'].apply(lambda x: x.replace('.' , '/' ))



# Showing the discrepancies in data.

print('Showing formats of date other than dd/mm/yyyy:')

for i in data['Date']:

    if not re.match(r'\b[0-9][0-9]/[0-9][0-9]/[0-9][0-9][0-9][0-9]' , i.strip()):

        print(i)

        

# As you can see incorrect formats for dates are d/mm/yyyy , dd/m/yyyy , d/m/yyyy or dd/mmyyyy.        

# We have to convert them into dd/mm/yyyy 



# Function to put all date values in correct format.

def correct_date(date):



    # Extract all the digits in form dd/mm/yyyy and store them in a list.

    if re.match(r'\b[0-9][0-9]/[0-1][0-9]/[0-9][0-9][0-9][0-9]' , date.strip()):

        digits = re.findall(r'\d' , date)

        digits.insert(2,'/')

        digits.insert(5,'/')

        

    # Extract all the digits in form d/mm/yyyy and store them in a list.    

    elif re.match(r'\b[0-9]/[0-1][0-9]/[0-9][0-9][0-9][0-9]' , date.strip()):

        digits = re.findall(r'\d' , date)

        digits.insert(0,'0')

        digits.insert(2,'/')

        digits.insert(5,'/')

        

    # Extract all the digits in form d/m/yyyy and store them in a list.    

    elif re.match(r'\b[0-9]/[0-9]/[0-9][0-9][0-9][0-9]' , date.strip()):

        digits = re.findall(r'\d' , date)

        digits.insert(0,'0')

        digits.insert(2,'0')

        digits.insert(2,'/')

        digits.insert(5,'/')

    

    # Extract all the digits in form dd/m/yyyy and store them in a list.    

    elif re.match(r'\b[0-9][0-9]/[0-9]/[0-9][0-9][0-9][0-9]' , date.strip()):

        digits = re.findall(r'\d' , date)

        digits.insert(2,'0')

        digits.insert(2,'/')

        digits.insert(5,'/')

        

     # Extract all the digits in form dd/mmyyyy and store them in a list.    

    elif re.match(r'\b[0-9][0-9]/[0-1][0-9][0-9][0-9][0-9][0-9]' , date.strip()):

        digits = re.findall(r'\d' , date)

        digits.insert(2,'/')

        digits.insert(5,'/')

    

    # Extract all the digits in form dd/mm/yyyy and store them in a list.

    elif re.match(r'\b[0-9][0-9]/[0-1][0-9]/[0-9][0-9][0-9]' , date.strip()):

        digits = re.findall(r'\d' , date)

        digits.insert(2,'/')

        digits.insert(5,'/')

        digits.insert(6,'2')

        

    date_temp = ''

#     print('date: ' , date)

    # Finally we form the date in correct format.

    for i in range(10):

        date_temp = date_temp + digits[i]

#     print('new date: ' , date_temp)    

    return date_temp    



# Since there are no null values in date we can go ahead and clean the data.

data['Date'] = data['Date'].apply(correct_date) 



# Convert the data type as datetime 64 

data['Date']=pd.to_datetime(data['Date'],format='%d/%m/%Y')
# Check for null values.

print('Null values in Startup names are: ' , data['Startup Name'].isnull().sum())

print('Null values in Investor names are: ' , data['Investors Name'].isnull().sum())



# Replace the null values with anonymous.

data['Investors Name'][data['Investors Name'].isnull()] = 'Undisclosed Investor'



# Convert all the values to upper case to avoid confusion.

data['Investors Name'] = data['Investors Name'].apply(lambda x: x.upper())



# Replace & with 'AND'

data['Investors Name'] = data['Investors Name'].apply(lambda x: x.replace('&','AND'))



# A lot of values have \\XA2 replace them with ''

data['Investors Name'] = data['Investors Name'].apply(lambda x: re.sub('\\\\[A-Z][A-Z][0-9]','',x))



# Now replace \ with ''

data['Investors Name'] = data['Investors Name'].apply(lambda x: re.sub('\\\\',' ',x))



# Remove any unnecessary spaces.

data['Investors Name'] = data['Investors Name'].apply(lambda x: x.replace('  ',''))



# Remove - 

data['Investors Name'] = data['Investors Name'].apply(lambda x: x.replace('-',''))



#Display the unique values in Investors Name.

data['Investors Name'].value_counts()
# Create a unique set.

set_in = create_unique_Set(data['Investors Name'])

#Find the longest match.

data['Investors Name'] = data['Investors Name'].apply(find_longest_match, set_values = set_in)



# Since we have a number of investors we need to make a new column to define the number of investors.

def no_of_investors(value):

    if re.search(',',value):

        return len(value.split(','))

    else:

        return 1

data['No of Investors'] = data['Investors Name'].apply(no_of_investors)
# Remove .com to avoid confusion.

data['Startup Name'] = data['Startup Name'].apply(lambda x: re.sub('.com',' ',x))



# Convert all the values to upper case to avoid confusion.

data['Startup Name'] = data['Startup Name'].apply(lambda x: x.upper())



# Replace & with 'AND'

data['Startup Name'] = data['Startup Name'].apply(lambda x: x.replace('&','AND'))



# A lot of values have \\XA2 replace them with ''

data['Startup Name'] = data['Startup Name'].apply(lambda x: re.sub('\\\\[A-Z][A-Z][0-9]','',x))



# Now replace \ with ''

data['Startup Name'] = data['Startup Name'].apply(lambda x: re.sub('\\\\',' ',x))



# Remove any unnecessary spaces.

data['Startup Name'] = data['Startup Name'].apply(lambda x: x.replace('  ',''))



# Remove - 

data['Startup Name'] = data['Startup Name'].apply(lambda x: x.replace('-',''))



#Display the unique values in Investors Name.

data['Startup Name'].value_counts()
# Create a unique set.

set_sn = create_unique_Set(data['Startup Name'])

#Find the longest match.

data['Startup Name'] = data['Startup Name'].apply(find_longest_match, set_values = set_sn)



#Display the cleaned data.

data['Startup Name'].value_counts()
# First we find out the monthly increase in data.



# Create a df to store the fund values.

col = ['Jan' , 'Feb' , 'Mar' , 'Apr' , 'May' , 'June' , 'July' , 'Aug' , 'Sep' , 'Oct' , 'Nov' , 'Dec']

fund_year = [2015,2016,2017,2018,2019]

fund_df = pd.DataFrame(columns=col , index=fund_year)



# Set the default values as zero

fund_df[fund_df[::].isnull()] = 0



fund_df_temp = pd.DataFrame(columns= ['Year' , 'Month' , 'Amount in USD'] )



# Store the month wise funding recieved in the df.    

for i,v in data.iterrows():

    mn = v['Date'].month

    yr = v['Date'].year

    fund_df.loc[yr][col[mn-1]] += v['Amount in USD']    

    fund_df_temp = fund_df_temp.append({'Year' : yr , 'Month' : mn , 'Amount in USD': v['Amount in USD'] } , ignore_index = True)
# Plot a line-plot to show the month-wise funding recieved.

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

ax = sns.relplot( height = 10,x = 'Month' , style = 'Year' ,kind = 'line', lw = 2, y = 'Amount in USD' , sort = col,  palette = ['black' , 'maroon' , 'navy' , 'limegreen' , 'mediumvioletred'], hue= 'Year', data = fund_df_temp)

plt.xticks(np.arange(1,13) , col , rotation = 45)

plt.title('Monthly analysis of Funding Recieved.')

plt.show()
# Plot a bar plot to show the year wise funding recieved by startups.

plt.figure(figsize= (12,12))

plt.ylabel('Amount in USD')         

plt.xlabel('Year')

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

sns.barplot(x = fund_df[::].index ,  y = [fund_df.loc[2015].values.sum() , fund_df.loc[2016].values.sum() , 

           fund_df.loc[2017].values.sum() , fund_df.loc[2018].values.sum() , fund_df.loc[2019].values.sum()] , palette="RdBu" 

          ) 

plt.title('Year-Wise Funding of startups.')

plt.show()

plt.figure(figsize=(12,12))

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

sns.barplot(y = data.groupby('Startup Name').sum().sort_values(by = 'Amount in USD' ,ascending = False)[:20]['Amount in USD'].index , x = 'Amount in USD' , data = data.groupby('Startup Name').sum().sort_values(by = 'Amount in USD' ,ascending = False)[:20]) 

plt.title('Top 20 Startups which recieved the maximum funding.')

plt.show()
plt.figure(figsize=(12,12))

plt.xticks(rotation = 90)

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

sns.barplot(x = data.groupby('Startup Name').count().sort_values(by = 'No of Investors' ,ascending = False)[:20]['Amount in USD'].index , palette='GnBu' , y = 'No of Investors' , data = data.groupby('Startup Name').sum().sort_values(by = 'No of Investors' ,ascending = False)[:20]) 

plt.title('Top 20 Startups with largest no of investors.' )

plt.show()
data.groupby('Investors Name').sum().sort_values(by = 'No of Investors' ,ascending = False)[:20]
# As we can see there are investors who made combined investment. 

# So, we try to find out the investors who made the maximum number of investment.



# Create a function to seperate the investors and store them 

investors_set = set()

def seperate_investors(series):



    for i in series.values:

        if re.search(',' , i):

            t_lst = i.split(',')

            for j in t_lst:

                investors_set.add(j)

        else:

            investors_set.add(i)

            

seperate_investors(data['Investors Name'])



# Now create a new dataframe.

investment_df = pd.DataFrame(columns=['Investor Name' , 'No of Investment'])

investment_df['No of Investment'] = investment_df['No of Investment'].astype('float') 

# Initialize the dataframe.

for i in investors_set:

    if i != '':

        investment_df = investment_df.append({'Investor Name':i , 'No of Investment':0} , ignore_index = True)

        

# Populate the dataframe.

for name in data['Investors Name']:

    if re.search(',', name):

        temp_lst = name.split(',')

        for nm in temp_lst:

            investment_df.loc[investment_df['Investor Name'] == nm , 'No of Investment'] += 1.0 

    else:

        investment_df.loc[investment_df['Investor Name'] == name , 'No of Investment']  += 1.0 

         

# investment_df[investment_df['Investor Name']== 'EQUITY CREST']['No of Investment'] = 3

investment_df.loc[investment_df['Investor Name']== 'EQUITY CREST' , 'No of Investment'] += 1            
investment_df.sort_values(by= 'No of Investment', ascending=False)[:20]['Investor Name']
# Plot the graph

plt.figure(figsize=(12,12))

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

sns.barplot(y =investment_df.sort_values(by= 'No of Investment', ascending=False)[:20]['Investor Name'] , x = 'No of Investment' , data = investment_df.sort_values(by= 'No of Investment', ascending=False)[:20] , palette= 'dark') 

plt.title('Top 20 Investors who invested the maximum number of times.' )

plt.show()

# Here we find the type of funding(top 5) recieved by startups.

font = {'family' : 'normal',

        'weight' : 'bold',

        'size'   : 15}

plt.rc('font', **font)

labels = data.groupby('Investment Type').sum().sort_values(by = 'Amount in USD' , ascending = False)[:5].index

values = data.groupby('Investment Type').sum().sort_values(by = 'Amount in USD' , ascending = False)[:5]['Amount in USD'].values

fig , ax = plt.subplots()

fig.set_size_inches(12,12)

ax.pie(colors = ['b' , 'g' , 'c' , 'm' , 'y'] ,  labels = labels , x = values , autopct='%.1f%%' , explode = [0.1 for x in range(5)])

plt.title(' Top five types of funding recieved by startups.' , fontsize = 20)

plt.show()
# Here we find the industry vertical which recieved the maximum funding.

font = {'family' : 'normal',

        'weight' : 'bold',

        'size'   : 22}

plt.rc('font', **font)

labels = data.groupby('Industry Vertical').sum().sort_values(by = 'Amount in USD' , ascending = False)[:10].index

values = data.groupby('Industry Vertical').sum().sort_values(by = 'Amount in USD' , ascending = False)[:10]['Amount in USD'].values

fig , ax = plt.subplots()

fig.set_size_inches(12,12)

ax.pie(  labels = labels , x = values , autopct='%.1f%%' , explode = [0.1 for x in range(10)])

plt.title(' Percentage of funding recieved by top ten industry verticals.' , fontsize = 30)

plt.show()
colors = ['#6987C2' ,'#947EB0' , '#A9D2D5' , '#ADFCF9' , '#4B644A' , '#2589BD' , '#E8AEB7' , '#58A4B0' , '#A9A587' ]

data.groupby('City  Location' ).sum().sort_values(by = 'Amount in USD' , ascending = False ).index
# We have to clean the data to avoid duplicity. 

data.loc[(data['City  Location'] == 'AHEMADABAD') | (data['City  Location'] == 'AHMEDABAD') , 'City  Location'] = 'AHMEDABAD'

data.loc[(data['City  Location'] == 'BANGALORE') | (data['City  Location'] == 'BENGALURU') , 'City  Location'] = 'BENGALURU'

data.loc[(data['City  Location'] == 'GURGAON') | (data['City  Location'] == 'GURUGRAM') , 'City  Location'] = 'GURUGRAM'



# Plot the data.

city_data = data.groupby('City  Location' ).sum().sort_values(by = 'Amount in USD' , ascending = False)[:10]

plt.figure(figsize=(12,12))

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

sns.barplot(y = city_data.index , x = 'Amount in USD' , data = city_data , palette= 'pastel') 

plt.title('Top 10 Cities which recieved the maximum funding.' )

plt.show()
# Plot the data.

city_data = data.groupby('City  Location' ).sum().sort_values(by = 'No of Investors' , ascending = False)[:10]

plt.figure(figsize=(12,12))

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

sns.barplot(y = city_data.index , x = 'No of Investors' , data = city_data , palette= 'BuGn') 

plt.title('Top 10 Cities which had the maximum number of investors.' )

plt.show()
top_cities = data.groupby('City  Location' ).count().sort_values(by = 'Startup Name' , ascending = False)[:10]

# Here we find the industry vertical which recieved the maximum funding.

font = {'family' : 'normal',

        'weight' : 'bold',

        'size'   : 15}

plt.rc('font', **font)

labels = top_cities.index

values = top_cities['Startup Name'].values

fig , ax = plt.subplots()

fig.set_size_inches(12,12)

ax.pie(  colors = colors,labels = labels , x = values , autopct='%.1f%%' , explode = [0.1 for x in range(10)])

plt.title(' Percentage of number of startups recieved by top ten cities.' , fontsize = 30)

plt.show()
