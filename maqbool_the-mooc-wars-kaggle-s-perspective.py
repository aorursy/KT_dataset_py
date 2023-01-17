### IMPORT LIBRARIES AND DATA ###
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Special setting for matplotlib to play nice with Jupyter
%matplotlib inline

# Read in the data
dat = pd.read_csv('../input/multipleChoiceResponses.csv', dtype='object')

### START OF DATA PREP ###
# Initial Data Preprocessing to make visualizations nicer
# Drop first row with question text
dat.drop(dat.index[0], inplace=True)

# Collapse undergrad category
dat['Q4'] = dat['Q4'].apply(lambda x: x if x != 'Some college/university study without earning a bachelor’s degree' else 'Bachelor’s degree')
# Shorten some country names
dat['Q3'] = dat['Q3'].apply(lambda x: x if x != 'United States of America' else 'USA')  
dat['Q3'] = dat['Q3'].apply(lambda x: x if x != 'United Kingdom of Great Britain and Northern Ireland' else 'UK')  

# Set categorical variables to be ordinal
dat['Age'] = dat['Q2'].astype('category').cat.as_ordered()                  # Age range
dat['Role_Experience'] = dat['Q8'].astype('category').cat.as_ordered()       # Experience in current role
dat['Coding%'] = dat['Q23'].astype('category').cat.as_ordered()              # Percentage of time coding
dat['Data_Experience'] = dat['Q24'].astype('category').cat.as_ordered()      # Years of experience with data analysis
dat['ML_Experience'] = dat['Q25'].astype('category').cat.as_ordered()       # Years of experience with machine learning 
dat['Bias_Exploration'] = dat['Q43'].astype('category').cat.as_ordered()    # Percentage of projects where model bias is explored
dat['Insight_Exploration'] = dat['Q46'].astype('category').cat.as_ordered()   # Percentage of projects where model insights are explored

# Rename columns
dat['Gender'] = dat['Q1']
dat['Country'] = dat['Q3']

# Convert numeric to float for easier manipulation and meaningful names
dat['Gather Data'] = dat['Q34_Part_1'].astype(float)
dat['Clean Data'] = dat['Q34_Part_2'].astype(float)
dat['Visualize Data'] = dat['Q34_Part_3'].astype(float)
dat['Model Build/Select'] = dat['Q34_Part_4'].astype(float)
dat['Deploy to Prod'] = dat['Q34_Part_5'].astype(float)
dat['Find Insights'] = dat['Q34_Part_6'].astype(float)
dat['Other Time'] = dat['Q34_OTHER_TEXT'].astype(float)

dat['MOOC_Time'] = dat['Q35_Part_2'].astype(float)


#### END OF DATA PREP ####


# Setup data for pie chart
mooc_use = dat['MOOC_Time'].describe()[1]
labels = ['Online Learning', 'Other Learning']
colors = ['#1DD040', '#E6E6E6']
explode = [0.1, 0]
sizes = [mooc_use, 100 - mooc_use]

# Plot pie chart
fig1, ax1 = plt.subplots(figsize=(4,4))
plt.pie(sizes, colors=colors, explode=explode, autopct='%1.0f%%',\
        textprops={'color': 'white', 'fontsize': 22});

# Add text
plt.text(1.1, 0.6, 'portion of learning\nKagglers do in', fontsize=14, color='grey')
plt.text(2.25, 0.59, 'MOOC', fontsize=14, color='#1DD040', fontweight='bold');
# Aggregate Kaggler's opinions on MOOC
cnts = dat['Q39_Part_1'].value_counts()
no_opinion = cnts['No opinion; I do not know']
cnts = cnts.drop(labels = ['No opinion; I do not know'])
cnts = cnts/cnts.sum()    # convert to percentage


# Plot
# Set order and colors
sns.set()
pref_order = ['Much worse', 'Slightly worse', 'Neither better nor worse', 'Slightly better', 'Much better']
pref_color = ['#F7819F', '#F5A9BC', '#E6E6E6', '#CEF6D8', '#9FF781']

# matplotlib general settings
fig, ax = plt.subplots(figsize=(20,1))
plt.title('Q39: How do you perceive the quality of MOOCs compared to traditional education?', fontsize=18, loc='left')
ax.get_xaxis().set_visible(False)
ax.tick_params(axis='y', labelsize=16, labelcolor='grey')  
ax.set_facecolor('white')

# Draw each bar and text separately with appropriate offset
bar_start = 0
for i in pref_order:
    ax.barh(y=['All Respondents'], width=cnts[i], height=0.1, left=bar_start, color=pref_color[pref_order.index(i)])
    plt.text(bar_start + cnts[i]/2 - 0.01, -0.01, "{:.0%}".format(cnts[i]), fontsize=16)
    #plt.text(bar_start + (cnts[i])/2 - 0.015, 0.4, "{:.0%}".format(cnts[i]), fontsize=16, transform=ax.transAxes)
    bar_start += cnts[i]

# Draw legend and set color of its text
leg = ax.legend(pref_order, loc=(0.18,-0.5), ncol=5, fontsize=14, frameon=True, facecolor='white');
for txt in leg.get_texts():
    plt.setp(txt, color='grey')

# Categories are age brackets from lowest to highest
categories = ['Doctoral degree', 'Master’s degree', 'Bachelor’s degree']

# Empty df to be built out
cnts = pd.DataFrame(columns = categories)

# Loop over all age categories and get distribution of responses 
for cat in categories:
    cnts[cat] = dat.loc[dat['Q4'] == cat, 'Q39_Part_1'].value_counts()

# Drop those with no opinion
cnts = cnts.drop('No opinion; I do not know')
cnts = cnts/cnts.sum()    # convert to percentage


# Plot

# matplotlib settings
fig, ax = plt.subplots(figsize=(20,3))
plt.title('Q39: How do you perceive the quality of MOOCs compared to traditional education?', fontsize=18, loc='left')
ax.get_xaxis().set_visible(False)
ax.tick_params(axis='y', labelsize=16, labelcolor='grey')  
ax.set_facecolor('white')

# Draw each bar and text separately with appropriate offset
for cat in categories:
    bar_start = 0
    for i in pref_order:
        ax.barh(y=[cat], width=cnts.loc[i,cat], height=0.6, left=bar_start, color=pref_color[pref_order.index(i)])
        plt.text(bar_start + cnts.loc[i,cat]/2 - 0.01, categories.index(cat) - 0.1, "{:.0%}".format(cnts.loc[i,cat]), fontsize=14)
        bar_start += cnts.loc[i,cat]

# Draw legend and set color of its text
leg = ax.legend(pref_order, loc=(0.18,-0.2), ncol=5, fontsize=14, frameon=True, facecolor='white');
for txt in leg.get_texts():
    plt.setp(txt, color='grey')
# Categories are top 3 countries based on number of responses
categories = ['USA', 'China', 'India']

# Empty df to be built out
cnts = pd.DataFrame(columns = categories)

# Loop over all age categories and get distribution of responses 
for cat in categories:
    cnts[cat] = dat.loc[dat['Country'] == cat, 'Q39_Part_1'].value_counts()

# Drop those with no opinion
cnts = cnts.drop('No opinion; I do not know')
cnts = cnts/cnts.sum()    # convert to percentage


# Plot

# matplotlib settings
fig, ax = plt.subplots(figsize=(20,3))
plt.title('Q39: How do you perceive the quality of MOOCs compared to traditional education?', fontsize=18, loc='left')
ax.get_xaxis().set_visible(False)
ax.tick_params(axis='y', labelsize=16, labelcolor='grey')  
ax.set_facecolor('white')

# Draw each bar and text separately with appropriate offset
for cat in categories:
    bar_start = 0
    for i in pref_order:
        ax.barh(y=[cat], width=cnts.loc[i,cat], height=0.6, left=bar_start, color=pref_color[pref_order.index(i)])
        if cnts.loc[i,cat] > 0.02:
            plt.text(bar_start + cnts.loc[i,cat]/2 - 0.01, categories.index(cat) - 0.1, "{:.0%}".format(cnts.loc[i,cat]), fontsize=14)
        bar_start += cnts.loc[i,cat]

# Draw legend and set color of its text
leg = ax.legend(pref_order, loc=(0.18,-0.2), ncol=5, fontsize=14, frameon=True, facecolor='white');
for txt in leg.get_texts():
    plt.setp(txt, color='grey')
# TECHNICAL NOTE: if a respondent only provided one answer in question 36 (which online learning platforms do you use), 
#                 then question 37 was skipped. It is assumed that the one platform from Q36 is the platform on which 
#                 the respondent spends most time. To take that into account I do some data prep before plottign the graph.

# Iterate over a slise of dataset (questions 36, all parts)
for i, row in dat.iloc[:,291:304].iterrows():
    if row['Q36_Part_12'] == 'None':
        dat.loc[i, 'Q37'] = 'None'                       # Responded doesn't use MOOCs
    elif row.count() == 1:
        dat.loc[i, 'Q37'] = ''.join(row.fillna(''))      # Respondent gave only one MOOC

## Count how many MOOC platforms the respondent uses
dat['No_MOOCs'] = dat.iloc[:,291:304].count(axis=1)

# Plot

# Get the counts of each MOOC
cnts = dat.loc[dat['No_MOOCs'] == 1, 'Q37'].value_counts()
cnts = cnts.drop(['None'])

# Assign different color to 'Other' category
plt_data = pd.DataFrame(cnts)
plt_data['hue'] = ['#CEF6D8'] * plt_data.shape[0]
plt_data.loc['Other', 'hue'] = '#E6E6E6'


# Basic plot setting
fig, ax = plt.subplots(figsize=(12,6))
ax.set_facecolor('white')
plt.barh(y=plt_data.index, width=plt_data.Q37, height=0.7, color=plt_data.hue)
plt.title('Single-platform MOOC Learner Preferences (Q37 inferred)', fontsize=14, loc='left')


# Remove figure frame and Y grid
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)


# X-axis formatting
plt.xlabel('Number of Respondents', labelpad = 10)
ax.tick_params(axis='x', colors='grey', labelsize=12)

# Y-axis formatting
ax.yaxis.grid(False)
ax.tick_params(axis='y', colors='black', labelsize=12)

# Plot from highest to lowest count
plt.gca().invert_yaxis()

# Draw callout box
rect = patches.Rectangle((0,-0.6),1595,6.08,linewidth=0.4,edgecolor='grey',facecolor='none')
plt.text(1030, 5.2, '80% of single-platform learners', fontsize=14, color='grey')
ax.add_patch(rect);
# Plot

# Get the counts of each MOOC
cnts = dat.loc[dat['No_MOOCs'] > 1, 'Q37'].value_counts()

# Assign different color to 'Other' category
plt_data = pd.DataFrame(cnts)
plt_data['hue'] = ['#CEF6D8'] * plt_data.shape[0]
plt_data.loc['Other', 'hue'] = '#E6E6E6'


# Basic plot setting
fig, ax = plt.subplots(figsize=(12,6))
ax.set_facecolor('white')
plt.barh(y=plt_data.index, width=plt_data.Q37, height=0.7, color=plt_data.hue)
plt.title('Q37: On which learning platform have you spent the most amount of time?', fontsize=14, loc='left')


# Remove figure frame and Y grid
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)


# X-axis formatting
plt.xlabel('Number of Respondents', labelpad = 10)
ax.tick_params(axis='x', colors='grey', labelsize=12)

# Y-axis formatting
ax.yaxis.grid(False)
ax.tick_params(axis='y', colors='black', labelsize=12)

# Plot from highest to lowest count
plt.gca().invert_yaxis()

# Draw callout box
rect = patches.Rectangle((0,-0.6),3800,5.08,linewidth=0.4,edgecolor='grey',facecolor='none')
plt.text(2480, 4.2, '80% of multi-platform learners', fontsize=14, color='grey')
ax.add_patch(rect);
# Select appropriate slices of respondents: those who used at least 2 platforms of interest
coursera = []
coursera.append(dat[(dat['Q36_Part_2'] == 'Coursera') & (dat['Q36_Part_4'] == 'DataCamp')]['Q37'].value_counts()[0])
coursera.append(dat[(dat['Q36_Part_2'] == 'Coursera') & (dat['Q36_Part_9'] == 'Udemy')]['Q37'].value_counts()[0])
coursera.append(dat[(dat['Q36_Part_2'] == 'Coursera') & (dat['Q36_Part_1'] == 'Udacity')]['Q37'].value_counts()[0])
coursera.append(dat[(dat['Q36_Part_2'] == 'Coursera') & (dat['Q36_Part_3'] == 'edX')]['Q37'].value_counts()[0])

competition = []
competition.append(dat[(dat['Q36_Part_2'] == 'Coursera') & (dat['Q36_Part_4'] == 'DataCamp')]['Q37'].value_counts()[1])
competition.append(dat[(dat['Q36_Part_2'] == 'Coursera') & (dat['Q36_Part_9'] == 'Udemy')]['Q37'].value_counts()[1])
competition.append(dat[(dat['Q36_Part_2'] == 'Coursera') & (dat['Q36_Part_1'] == 'Udacity')]['Q37'].value_counts()[1])
competition.append(dat[(dat['Q36_Part_2'] == 'Coursera') & (dat['Q36_Part_3'] == 'edX')]['Q37'].value_counts()[1])

# Format data for plotting
plt_data = pd.DataFrame({'Coursera': coursera, 'Competition': competition})
plt_data = (plt_data.transpose()/plt_data.sum(axis=1)).transpose()

competition_labels = ['DataCamp', 'Udemy', 'Udacity', 'edX']

fig, ax = plt.subplots(figsize=(12,4))
plt.tight_layout()
ax.get_xaxis().set_visible(False)
ax.set_facecolor('white')

plt.barh(width = plt_data['Competition'], y=competition_labels, color = '#E6E6E6', height=0.65)
plt.barh(width = plt_data['Coursera'], left=plt_data['Competition'], y=competition_labels, height=0.65, color='#0068B0')

# Add line markers
plt.plot([0.5, 0.5], [-0.6, 3.3], color='#6E6E6E', linestyle='--', linewidth=1)
plt.plot([0.25, 0.25], [-0.6, 3.3], color='#6E6E6E', linestyle='--', linewidth=1)

# Add text labels
plt.text(0, 4, 'What % of times, Kagglers choose Coursera over competition? (Q36 and Q37 inferred)', fontsize=14)
plt.text(0.91, 3.5, 'Coursera', color='#0068B0', fontsize=13, fontweight='bold')
plt.text(0, 3.5, 'Competition', color='#6E6E6E', fontsize=13, fontweight='bold')
plt.text(0.49, 3.5, 'vs.', color='#6E6E6E', fontsize=12, fontweight='bold')
plt.text(0.47, -0.8, 'Tie = 50%', color='#6E6E6E', fontsize=11)
plt.text(0.18, -0.8, 'Winning = 75%', color='#6E6E6E', fontsize=11);

ax.tick_params(axis='y', colors='grey', labelsize=14)
# Select appropriate slices of respondents: those who used at least 2 platforms of interest
datacamp = []
datacamp.append(dat[(dat['Q36_Part_4'] == 'DataCamp') & (dat['Q36_Part_6'] == 'Kaggle Learn')]['Q37'].value_counts()[0])
datacamp.append(dat[(dat['Q36_Part_4'] == 'DataCamp') & (dat['Q36_Part_9'] == 'Udemy')]['Q37'].value_counts()[0])
datacamp.append(dat[(dat['Q36_Part_4'] == 'DataCamp') & (dat['Q36_Part_3'] == 'edX')]['Q37'].value_counts()[0])
datacamp.append(dat[(dat['Q36_Part_4'] == 'DataCamp') & (dat['Q36_Part_1'] == 'Udacity')]['Q37'].value_counts()[0])

competition = []
competition.append(dat[(dat['Q36_Part_4'] == 'DataCamp') & (dat['Q36_Part_6'] == 'Kaggle Learn')]['Q37'].value_counts()[1])
competition.append(dat[(dat['Q36_Part_4'] == 'DataCamp') & (dat['Q36_Part_9'] == 'Udemy')]['Q37'].value_counts()[1])
competition.append(dat[(dat['Q36_Part_4'] == 'DataCamp') & (dat['Q36_Part_3'] == 'edX')]['Q37'].value_counts()[1])
competition.append(dat[(dat['Q36_Part_4'] == 'DataCamp') & (dat['Q36_Part_1'] == 'Udacity')]['Q37'].value_counts()[1])

# Format data for plotting
plt_data = pd.DataFrame({'DataCamp': datacamp, 'Competition': competition})
plt_data = (plt_data.transpose()/plt_data.sum(axis=1)).transpose()

competition_labels = ['Kaggle Learn', 'Udacity', 'edX', 'Udemy']

fig, ax = plt.subplots(figsize=(12,4))
plt.tight_layout()
ax.get_xaxis().set_visible(False)
ax.set_facecolor('white')

plt.barh(width = plt_data['Competition'], y=competition_labels, color = '#E6E6E6', height=0.65)
plt.barh(width = plt_data['DataCamp'], left=plt_data['Competition'], y=competition_labels, height=0.65, color='#3BB3D2')

# Add line markers
plt.plot([0.5, 0.5], [-0.6, 3.3], color='#6E6E6E', linestyle='--', linewidth=1)
plt.plot([0.25, 0.25], [-0.6, 3.3], color='#6E6E6E', linestyle='--', linewidth=1)

# Add text labels
plt.text(0, 4, 'What % of times, Kagglers choose DataCamp over competition? (Q36 and Q37 inferred)', fontsize=14)
plt.text(0.9, 3.5, 'DataCamp', color='#3BB3D2', fontsize=13, fontweight='bold')
plt.text(0, 3.5, 'Competition', color='#6E6E6E', fontsize=13, fontweight='bold')
plt.text(0.49, 3.5, 'vs.', color='#6E6E6E', fontsize=12, fontweight='bold')
plt.text(0.47, -0.8, 'Tie = 50%', color='#6E6E6E', fontsize=11)
plt.text(0.18, -0.8, 'Winning = 75%', color='#6E6E6E', fontsize=11);

ax.tick_params(axis='y', colors='grey', labelsize=14)
# Read in the data
txt_dat = pd.read_csv('../input/freeFormResponses.csv', dtype='object')

# Group platforms that appear under different names and/or are misspelled
def txt_process(txt):
    if 'linkedin' in txt or 'lynda' in txt or 'linda' in txt or 'lybda' in txt:
        return 'Linkedin / lynda.com'
    elif 'codeacademy' in txt or 'codecademy' in txt or 'code academy' in txt:
        return 'codecademy'
    elif 'cognitive' in txt:
        return 'cognitive class ai'
    elif 'mlcourse' in txt:
        return 'mlcourse.ai'
    elif 'stepic' in txt or 'stepik' in txt:
        return 'stepik'
    elif 'nptel' in txt:
        return 'nptel'
    elif 'vidhya' in txt:
        return 'analytics vidhya'
    elif 'ods' in txt:
        return 'ods.ai'
    elif 'pluralsight' in txt:
        return 'pluralsight'
    else:
        return txt

# Process free text for question 36
txt_dat['Q36_OTHER_TEXT'] = txt_dat['Q36_OTHER_TEXT'].dropna().apply(lambda x: x.lower().strip())
txt_dat['Q36_OTHER_TEXT'] = txt_dat['Q36_OTHER_TEXT'].dropna().apply(txt_process)

# Plot

# Basic plot setting
fig, ax = plt.subplots(figsize=(12,6))
ax.set_facecolor('white')
plt.title('Q37: Other MOOC Platforms With More Than 10 Respondents', fontsize=14, loc='left')

# Horizontal bars
plt.barh(width=txt_dat['Q36_OTHER_TEXT'].value_counts()[:10], y=txt_dat['Q36_OTHER_TEXT'].value_counts()[:10].index, height=0.7, color='#CEF6D8');

# X-axis formatting
plt.xlabel('Number of Respondents', labelpad = 10)
ax.tick_params(axis='x', colors='grey', labelsize=12)

# Y-axis formatting
ax.yaxis.grid(False)
ax.tick_params(axis='y', colors='black', labelsize=12)

# Fix the order to be highest to lowest
plt.gca().invert_yaxis();