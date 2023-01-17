# As always, first some libraries
# Numpy handles matrices
import numpy as np
# Pandas handles data 
import pandas as pd
# Matplotlib is a plotting library
import matplotlib.pyplot as plt
# Set matplotlib to render imediately
%matplotlib inline
# Seaborn is a plotting library built on top of matplotlib that can handle some more advanced plotting
import seaborn as sns
# Define colors for seaborn
five_thirty_eight = [
    "#30a2da",
    "#fc4f30",
    "#e5ae38",
    "#6d904f",
    "#8b8b8b",
]
# Tell seaborn to use the 538 colors
sns.set(palette=five_thirty_eight)
# Load data with pandas
df = pd.read_csv('../input/balanced_bank.csv',index_col=0)
# Display first five rows for a rough overview
df.head()
# Count missing values per column
df.isnull().sum()
# Display datatypes
df.dtypes
# Function to draw frequencies by outcome
def draw_conditional_barplot(feature,df):
    # Set matplotlib style
    plt.style.use('fivethirtyeight')
    # Count the total yes responses in our dataset
    n_yes = len(df[df.y == 'yes'])
    # Count the total no responses in our dataset
    n_no = len(df[df.y == 'no'])
    # Count the frequencies of the different jobs for the yes people
    yes_cnts = df[df.y == 'yes'][feature].value_counts() / n_yes * 100
    # Count frequencies of jobs for the nay sayers
    no_cnts = df[df.y == 'no'][feature].value_counts() / n_no * 100
    
    # A potential problem of creating two different frequency tables is that if one group (perhaps the yes crowd)...
    # ... does not include a certain category (like a certain job) then it will not be in the frequency tables at all
    # ... So we have to join them in one table to ensure that all categories are included
    # ... When merging the frequncy tables, missing categories will be marked with 'NA'
    # ... We can then replace all NAs with zeros, which is the correct frequency
    
    # Create a new dataframe that includes all frequencies
    res = pd.concat([yes_cnts,no_cnts],axis=1)
    # Name the columns of the new dataframe (yes crowd and nay sayers)
    res.columns = ['yes','no']
    # Fill empty fields with zeros
    res = res.fillna(0)
    
    # N = number of categories
    N = len(res['yes'])
    # Create an array for the locations of the group (creates an array [0,1,2,...,N])
    ind = np.arange(N) 
    # Specify width of bars
    width = 0.35   
    # Create empty matplotlib plot
    fig, ax = plt.subplots()
    # Add bars of the nay sayers
    rects1 = ax.bar(ind, res['no'], width)
    # Add bars of the yes crowd
    rects2 = ax.bar(ind + width, res['yes'], width)

    # Add label: feature name (e.g. job) in percent
    ax.set_ylabel(feature + ' in percent')
    # Add title
    ax.set_title(feature + ' by conversion')
    # Add ticks 
    ax.set_xticks(ind + width / 2)
    # Add categorie names as tick labels
    ax.set_xticklabels(res.index.values)
    # Rotate labels 90 degrees
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=90)

    # Add legend
    ax.legend((rects1[0], rects2[0]), ('No', 'Yes'))
    
    # Render plot
    plt.show()
draw_conditional_barplot('job',df)
# Add job_ to every value in jobs so that the dummies have readable names
df['job'] = 'job_' + df['job'].astype(str)

# Get dummies
dummies = pd.get_dummies(df['job'])

# Add dummies to dataframe
df = pd.concat([df,dummies],axis=1)

# Remove original job column
del df['job']
df.dtypes
# Function to draw conditional distribution plot splot by whether the customer subscribed
def draw_conditional_distplot(feature,df):
    # Seat seaborn to use nice colors
    sns.set(palette=five_thirty_eight)
    # Draw the no plot
    sns.distplot(df[df.y == 'no'][feature],label='no')
    # Draw the yes plot
    sns.distplot(df[df.y == 'yes'][feature],label='yes')
    # Draw the legend
    plt.legend()
    # Display the plot
    plt.show()
# Now we can just use the function defined above
draw_conditional_distplot('age',df)
# Create old people group
df['age_old'] = np.where(df['age'] >= 60, 1,0)
# Create mid age people group
df['age_mid'] = np.where((df['age'] <= 60) & (df['age'] >= 35), 1,0)
# Create young people group
df['age_young'] = np.where(df['age'] <= 35, 1,0)
# Remove original age
del df['age']
draw_conditional_barplot('marital',df)
# Add marital_ to every value in marital so that the dummies have readable names
df['marital'] = 'marital_' + df['marital'].astype(str)

# Get dummies
dummies = pd.get_dummies(df['marital'])

# Add dummies to df
df = pd.concat([df,dummies],axis=1)

#remove original column
del df['marital']
draw_conditional_barplot('education',df)
# Add education_ to every value in education so that the dummies have readable names
df['education'] = 'education_' + df['education'].astype(str)

# Get dummies
dummies = pd.get_dummies(df['education'])

# Add dummies to df
df = pd.concat([df,dummies],axis=1)

#remove original column
del df['education']
draw_conditional_barplot('default',df)
# This code was copied from our barchart plotting function, but we will print out the numbers instead of plotting them
# Count total yes responses
n_yes = len(df[df.y == 'yes'])
# Count the total no responses in our dataset
n_no = len(df[df.y == 'no'])
# Count the frequencies of the different jobs for the yes people
yes_cnts = df[df.y == 'yes']['default'].value_counts() / n_yes * 100
# Count frequencies of jobs for the nay sayers
no_cnts = df[df.y == 'no']['default'].value_counts() / n_no * 100

# Create a new dataframe that includes all frequencies
res = pd.concat([yes_cnts,no_cnts],axis=1)
# Name the columns of the new dataframe (yes crowd and nay sayers)
res.columns = ['yes','no']
# Fill empty fields with zeros
res = res.fillna(0)

print(res)
# Ensure dummy names are well readable
df['default'] = 'default_' + df['default'].astype(str)
# Get dummies
dummies = pd.get_dummies(df['default'])
# Get dummies
dummies = pd.get_dummies(df['default'])
# Add dummies to df
df = pd.concat([df,dummies],axis=1)
#remove original column
del df['default']
draw_conditional_barplot('housing',df)
# Ensure dummy names are well readable
df['housing'] = 'housing_' + df['housing'].astype(str)

# Get dummies
dummies = pd.get_dummies(df['housing'])

# Add dummies to df
df = pd.concat([df,dummies],axis=1)

#remove original column
del df['housing']
draw_conditional_barplot('loan',df)
# Ensure dummy names are well readable
df['loan'] = 'loan_' + df['loan'].astype(str)

# Get dummies
dummies = pd.get_dummies(df['loan'])

# Add dummies to df
df = pd.concat([df,dummies],axis=1)

#remove original column
del df['loan']
draw_conditional_barplot('contact',df)
# Ensure dummy names are well readable
df['contact'] = 'contact_' + df['contact'].astype(str)

# Get dummies
dummies = pd.get_dummies(df['contact'])

# Add dummies to df
df = pd.concat([df,dummies],axis=1)

#remove original column
del df['contact']
draw_conditional_barplot('month',df)
# Ensure dummy names are well readable
df['month'] = 'month_' + df['month'].astype(str)

# Get dummies
dummies = pd.get_dummies(df['month'])

# Add dummies to df
df = pd.concat([df,dummies],axis=1)

#remove original column
del df['month']
draw_conditional_barplot('day_of_week',df)
# Add job_ to every value in jobs so that the dummies have readable names
df['day_of_week'] = 'day_of_week_' + df['day_of_week'].astype(str)

# Get dummies
dummies = pd.get_dummies(df['day_of_week'])

# Add dummies to df
df = pd.concat([df,dummies],axis=1)

#remove original column
del df['day_of_week']
draw_conditional_barplot('poutcome',df)
# Add job_ to every value in jobs so that the dummies have readable names
df['poutcome'] = 'poutcome_' + df['poutcome'].astype(str)

# Get dummies
dummies = pd.get_dummies(df['poutcome'])

# Add dummies to df
df = pd.concat([df,dummies],axis=1)

#remove original column
del df['poutcome']
draw_conditional_distplot('duration',df)
del df['duration']
draw_conditional_distplot('campaign',df)
draw_conditional_distplot('campaign',df[df.campaign < 10])
df['campaign'] = df['campaign'].clip(upper = 10)
df['campaign'] = (df['campaign'] - df['campaign'].mean())/(df['campaign'].std())
draw_conditional_distplot('pdays',df[df['pdays'] < 500])
df['contacted_before'] = np.where(df['pdays'] == 999, 0,1)
draw_conditional_barplot('contacted_before',df)
df['pdays'] = np.where(df['pdays'] == 999,df[df['pdays'] < 999]['pdays'].mean(),df['pdays'])
draw_conditional_distplot('pdays',df)
df['pdays'] = (df['pdays'] - df['pdays'].mean())/(df['pdays'].std())
draw_conditional_distplot('previous',df)
df['previous'] = (df['previous'] - df['previous'].mean())/(df['previous'].std())
draw_conditional_distplot('emp.var.rate',df)
# Feature scaling
df['emp.var.rate'] = (df['emp.var.rate'] - df['emp.var.rate'].mean())/(df['emp.var.rate'].std())
draw_conditional_distplot('cons.price.idx',df)
# Feature scaling
df['cons.price.idx'] = (df['cons.price.idx'] - df['cons.price.idx'].mean())/(df['cons.price.idx'].std())
draw_conditional_distplot('cons.conf.idx',df)
# Feature scaling
df['cons.conf.idx'] = (df['cons.conf.idx'] - df['cons.conf.idx'].mean())/(df['cons.conf.idx'].std())
draw_conditional_distplot('euribor3m',df)
df['euribor3m'] = (df['euribor3m'] - df['euribor3m'].mean())/(df['euribor3m'].std())
draw_conditional_distplot('nr.employed',df)
# Feature scaling
df['nr.employed'] = (df['nr.employed'] - df['nr.employed'].mean())/(df['nr.employed'].std())
cnts = df.y.value_counts()
cnts.plot(kind='bar')
# Conversion
df['y'] = np.where(df['y'] == 'yes', 1,0)
pd.set_option('display.max_rows', 100)
df.dtypes
df.to_csv('processed_bank.csv')