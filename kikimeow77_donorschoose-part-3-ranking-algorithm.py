import numpy as np
import pandas as pd 
import os
import datetime as dt
from sklearn import cluster
import seaborn as sns
import matplotlib.pyplot as plt
import plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
py.offline.init_notebook_mode(connected=True)
%matplotlib inline
# define columns to import
projectCols = ['Project ID', 'School ID', 'Teacher ID',
               'Teacher Project Posted Sequence', 'Project Type',
               'Project Subject Category Tree', 'Project Subject Subcategory Tree',
               'Project Grade Level Category', 'Project Resource Category',
               'Project Cost', 'Project Posted Date', 'Project Expiration Date',
               'Project Current Status', 'Project Fully Funded Date']

resourcesCols = ['Project ID','Resource Quantity','Resource Unit Price', 'Resource Vendor Name']

# import files
donations = pd.read_csv('../input/io/Donations.csv', dtype = {'Donation Amount': np.float32, 'Donor Cart Sequence': np.int32})
donors = pd.read_csv('../input/io/Donors.csv', dtype = {'Donor Zip':'str'})
projects = pd.read_csv('../input/io/Projects.csv', usecols = projectCols, dtype = {'Teacher Project Posted Sequence': np.float32, 'Project Cost': np.float32})
resources = pd.read_csv('../input/io/Resources.csv', usecols = resourcesCols, dtype = {'Resource Quantity': np.float32,'Resource Unit Price': np.float32})
schools = pd.read_csv('../input/io/Schools.csv', dtype = {'School Zip': 'str'})
teachers = pd.read_csv('../input/io/Teachers.csv')

# These are files from Part I:
donorFeatureMatrixNoAdj = pd.read_csv('../input/part-1-preprocessing-feature-engineering/donorFeatureMatrixNoAdj.csv')
donorFeatureMatrix =  pd.read_csv('../input/part-1-preprocessing-feature-engineering/donorFeatureMatrix.csv')
donorsMapping = pd.read_csv('../input/part-1-preprocessing-feature-engineering/donorsMapping.csv') 
schoolsMapping = pd.read_csv('../input/part-1-preprocessing-feature-engineering/schoolsMapping.csv')
projFeatures = pd.read_csv('../input/part-1-preprocessing-feature-engineering/projFeatures.csv')
distFeatures = pd.read_csv('../input/part-1-preprocessing-feature-engineering/distFeatures.csv')

# donations
donations['Donation Received Date'] = pd.to_datetime(donations['Donation Received Date'])
donations['Donation Included Optional Donation'].replace(('Yes', 'No'), (1, 0), inplace=True)
donations['Donation Included Optional Donation'] = donations['Donation Included Optional Donation'].astype('bool')
donations['Donation_Received_Year'] = donations['Donation Received Date'].dt.year
donations['Donation_Received_Month'] = donations['Donation Received Date'].dt.month
donations['Donation_Received_Day'] = donations['Donation Received Date'].dt.day

# donors
donors['Donor Is Teacher'].replace(('Yes', 'No'), (1, 0), inplace=True)
donors['Donor Is Teacher'] = donors['Donor Is Teacher'].astype('bool')

# projects
cols = ['Project Posted Date', 'Project Fully Funded Date']
projects.loc[:, cols] = projects.loc[:, cols].apply(pd.to_datetime)
projects['Days_to_Fullyfunded'] = projects['Project Fully Funded Date'] - projects['Project Posted Date']

# teachers
teachers['Teacher First Project Posted Date'] = pd.to_datetime(teachers['Teacher First Project Posted Date'])

##
# Name the dataframes
##
def name_dataframes(dfList, dfNames):
    '''
    give names to a list of dataframes. 
    Argument:
        dfList = list of dataframes,
        dfNames = list of names for the dataframes
    Return:
        None
    '''
    for df, name in zip(dfList, dfNames):
        df.name = name
    
    return

dfList = [donations, donors, projects, resources, schools, teachers]
dfNames = ['donations', 'donors', 'projects', 'resources', 'schools', 'teachers']
name_dataframes(dfList, dfNames)

##
#  Remove rows in the datasets that cannot be mapped
##
projects = projects.loc[projects['School ID'].isin(schools['School ID'])]
projects = projects.loc[projects['Project ID'].isin(resources['Project ID'])]
donations = donations.loc[donations['Project ID'].isin(projects['Project ID'])]
donations = donations.loc[donations['Donor ID'].isin(donors['Donor ID'])]
donors = donors.loc[donors['Donor ID'].isin(donations['Donor ID'])]

##
#  We will add features we created in Part I to the donors and schools dataset.
##
donors = donors.merge(donorsMapping, left_on = 'Donor ID', right_on = 'Donor ID', how = 'left')
donors = donors.drop(['Unnamed: 0'], axis=1)
schools = schools.merge(schoolsMapping.filter(items = ['School ID', 'School_Lon', 'School_Lat']), left_on = 'School ID', right_on = 'School ID', how = 'left')

projFeatures = projFeatures.drop(['Unnamed: 0'], axis=1)
def donor_summary(donations, projects):
    ''' 
    Generate features to refelect donor's previous donation history:
    'num_proj', 'num_donation', 'num_cart', 'donation_median',
    'donation_mean', 'donation_sum', 'donation_std', 'School ID_count',
    'Teacher ID_count', 'schoolConcentration', 'TeacherConcentration',
    '''
    donations = donations.set_index('Donor ID', drop = False)
    donorSummary = pd.DataFrame(donations['Donor ID']).drop_duplicates(keep = 'first') 
    
    #### Obtain number of projects, # of donations, and the max cart number for each  donor
    countProj = donations.groupby(donations.index).agg({'Project ID':'nunique','Donor Cart Sequence':'max', 'Donation ID':'count'})
    countProj.columns = ['num_proj', 'num_donation','num_cart']
    donorSummary  = donorSummary.merge(countProj, left_index = True,  right_index=True, how = 'left')
    
    #### Count # of schools and # of teachers that a donor donates to
    school_teacher = donations[['Project ID', 'Donation Amount', 'Donor ID']].merge(projects[['Project ID', 'School ID', 'Teacher ID']], left_on = 'Project ID', right_on = 'Project ID', how = 'left')
    concentration = school_teacher.groupby('Donor ID').agg({'School ID':'nunique', 'Teacher ID':'nunique'})
    concentration.columns = concentration.columns + '_count'
    donorSummary  = donorSummary.merge(concentration, left_index = True,  right_index=True, how = 'left')
    
    #### Design feature to capture the concentration of donation to schools.
    #### feature that captures doners that donates to multiple schools, and not just have one favorite school
    schoolSum = school_teacher.groupby(['Donor ID', 'School ID'])['Donation Amount'].sum().reset_index(drop = False)
    schoolSum = schoolSum.groupby(['Donor ID'])['Donation Amount'].agg(['sum', 'max'])
    schoolSum['SchoolConcentration'] = schoolSum['max']/schoolSum['sum']
    donorSummary['schoolConcentration'] = schoolSum['SchoolConcentration']
    
    #### Design feature to capture the concentration of donation to a teacher.  
    TeacherSum = school_teacher.groupby(['Donor ID', 'Teacher ID'])['Donation Amount'].sum().reset_index(drop = False)
    TeacherSum = TeacherSum.groupby(['Donor ID'])['Donation Amount'].agg(['sum', 'max'])
    TeacherSum['TeacherConcentration'] = TeacherSum['max']/TeacherSum['sum']
    donorSummary['TeacherConcentration'] = TeacherSum['TeacherConcentration']
    
    return donorSummary
donorSummary = donor_summary(donations, projects)
chartData = donorSummary.loc[donorSummary['num_proj']>1]
chartData['School Bias'] = chartData['School ID_count'] < chartData['num_proj']
Breakdown = chartData['School Bias'].value_counts()/len(chartData)
Breakdown.plot(kind='bar', stacked=True, legend = True)
plt.title("Most donors have donated to the same school")
Breakdown
chartData['1/NumProj'] = 1/chartData['num_proj']
chartData = chartData.sample(1000)
sns.lmplot(data = chartData, x= 'schoolConcentration', y= '1/NumProj', hue = 'School Bias', fit_reg=True)
plt.xlabel("% donation donated to the favorite school") 
plt.ylabel("1 / Number of Projects")
plt.title("Donors have Favorite Schools")
plt.ylim(0, 1)
plt.xlim(0, 1)
x1, y1 = [0, 1], [0, 1]
plt.plot(x1, y1, dashes=[6, 2], color = "pink")
chartData = donorSummary.loc[donorSummary['num_proj']>1]
chartData['Teacher Bias'] = (chartData['School ID_count']) > chartData['Teacher ID_count']
Breakdown = chartData['Teacher Bias'].value_counts()/len(chartData)
Breakdown.plot(kind='bar', stacked=True, legend = True)
plt.title('Most donors are not biased towards a specific teacher')
print(Breakdown)

chartData = chartData.sample(5000)
sns.lmplot(data = chartData, x= 'schoolConcentration', y= 'TeacherConcentration', hue = 'Teacher Bias', fit_reg=False)
plt.xlabel("% donation donated to the favorite school") 
plt.ylabel("% donation donated to the favorite teacher")
plt.title("Few donors are biased towards specific teachers")
plt.ylim(0, 1)
plt.xlim(0, 1)
x1, y1 = [0, 1], [0, 1]
plt.plot(x1, y1)
# Prepare Chart Data
chartData = donations[['Donor ID', 'Project ID', 'Donation Amount']].merge(distFeatures[['Donor ID', 'Project ID', 'dist']], left_on = ['Donor ID', 'Project ID'], right_on = ['Donor ID', 'Project ID'], how = 'left')
chartData = chartData.merge(donors[['Donor ID', 'no_mismatch']], left_on = 'Donor ID', right_on = 'Donor ID', how = 'left')
chartData = chartData.loc[chartData['no_mismatch'] == 1]
chartData['dist_cut'] = pd.cut(chartData['dist'], bins = [-1, 0, 5, 10, 20, 50, 100, 15000], labels = ['0', '1-5', '6-10', '11-20', '21-50', '51-100', '>100'])
chartData['Total Amount'] = chartData.groupby(['Donor ID', 'Project ID'])['Donation Amount'].transform('sum')
chartData = chartData.drop_duplicates(subset=['Project ID', 'Donor ID'], keep='first')
chart = pd.DataFrame(chartData.groupby('dist_cut')['Total Amount'].agg('sum')/chartData['Donation Amount'].sum()*100).reset_index(drop = False)

# Plot chart
plot = sns.barplot(x = 'dist_cut', y = 'Total Amount', data = chart)
plot.set_title("% of donations relative to distances between donor and schools")
plot.set_ylabel('% donations')
plot.set_xlabel('distances in miles')

print("The median distance in miles between project/donor pair is:", chartData['dist'].median())
print(chart)
# Run K-means model on donor FeatureMatrix 
k_means = cluster.KMeans(n_clusters=5)

# Group using project categories
colsInclude = list(donorFeatureMatrix.loc[:,'ProjCat_Applied Learning': 'ProjCat_Warmth, Care & Hunger'].columns)
result = k_means.fit(donorFeatureMatrix[colsInclude])

# Get the k-means grouping label
clusterLabel = result.labels_
pd.DataFrame(clusterLabel)[0].value_counts(normalize=True)
def plot_cluster_traits(donorFeatureMatrix, col_category, clusterLabel):
    '''
    col_category are the filters for the column names in the donorFeatureMatrix
    values could be: 
    'Project Type', 'School Metro Type', 'Project Grade Level Category',
    'Project Resource Category', 'lunchAid', 'ProjCat', 'Dist', 'Percentile'
    
    clusterLabel is labels from the output of k-means
    '''
    
    # get columns to chart
    chart = donorFeatureMatrix.filter(regex='^'+col_category, axis=1).copy()
    chart['label'] = clusterLabel
    
    # for each column, get mean of each cluster
    chart = chart.groupby(['label']).mean().reset_index()
    chart_melt = pd.melt(chart, id_vars = ['label'], value_vars = chart.columns[1:], var_name='category', value_name = 'mean')
    chart_melt['color'] = np.where(chart_melt['mean']<0, 'orange', 'pink')
    chart_melt = chart_melt.sort_values(by = ['label', 'category']).reset_index(drop = True)
    
    # delete the col_category from column names for the chart
    chart_melt['category'] = chart_melt['category'].str.replace(col_category+'_','')
    
    # plot chart using Seaborn
    if chart_melt['category'].nunique()>8:
        g = sns.FacetGrid(chart_melt, row = 'label', size=1.5, aspect=8)  # size: height, # aspect * size gives the width
        g.map(sns.barplot, 'category', 'mean', palette="Set1")
        g.set_xticklabels(rotation=90)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle('Cluster Preferences- ' + col_category)
    else:
        g = sns.FacetGrid(chart_melt, row = 'label', size=1.5, aspect=4)
        g.map(sns.barplot, 'category', 'mean', palette="Set2")
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle('Cluster Preferences- ' + col_category)
    return g
plot_cluster_traits(donorFeatureMatrix, 'ProjCat', clusterLabel)
donorFeatureMatrix.columns
donorID_2018 = donations[(donations['Donation_Received_Year'] == 2018) & (donations['Donation_Received_Month'] == 1)]['Donor ID'].unique()
donorID_prior_2018 = donations[donations['Donation_Received_Year'] < 2018]['Donor ID'].unique()
test_ID = list(set(donorID_2018).intersection(donorID_prior_2018))
print("# of IDs matching the criterias:", len(test_ID))
donations_prior = donations[(donations['Donation_Received_Year'] < 2018) & (donations['Donor ID'].isin(test_ID))]
donations_2018 = donations[(donations['Donation_Received_Year'] == 2018) & (donations['Donation_Received_Month'] == 1) & (donations['Donor ID'].isin(test_ID))]
print('donations_prior:', donations_prior.shape)
print('donations_Jan 2018:', donations_2018.shape)
donorFeatureMatrixNoAdj = donorFeatureMatrixNoAdj.loc[donorFeatureMatrixNoAdj['Donor ID'].isin(test_ID)]
donorFeatureMatrix = donorFeatureMatrix.loc[donorFeatureMatrix['Donor ID'].isin(test_ID)]
print('Donor Feature Matrix Unscaled Shape:', donorFeatureMatrixNoAdj.shape)
print('Donor Feature Matrix Scaled Shape:', donorFeatureMatrix.shape)
projectsID = set(projects[(projects['Project Fully Funded Date'] >= '1/1/2018') & (projects['Project Expiration Date'] >= '1/1/2018') & (projects['Project Posted Date'] <= '1/31/2018')]['Project ID'])
print("# of Project IDs matching the criterias:", len(projectsID))
projects_2018 = list(donations[(donations['Donation_Received_Year'] == 2018) & (donations['Donation_Received_Month'] == 1)]['Project ID'].unique())
print("# of projects that people donated to in Jan 2018:", len(projects_2018))

projectsID = set(projects_2018).union(set(projectsID))
print("# of projects that people could donate to in Jan 2018:", len(projectsID))
projectFeatures = projFeatures.loc[projFeatures['Project ID'].isin(projectsID)]
projectFeatures = projectFeatures.merge(projects.filter(items = ['Project ID', 'School ID', 'Teacher ID']), left_on = 'Project ID', right_on = 'Project ID', how = 'left')
projectFeatures = projectFeatures.merge(schools.filter(items = ['School ID', 'School_Lon', 'School_Lat', 'School City', 'School State']), left_on = 'School ID', right_on = 'School ID', how = 'left')
projectFeatures = projectFeatures.drop_duplicates(subset=['Project ID'], keep='first')
projectFeatures['CityState'] = projectFeatures['School City'].map(str)+', '+projectFeatures['School State']
print("Number of projects in projectFeatures:", len(projectFeatures))
def merge_data(ID, donations, donors = donors, projects = projects, schools = schools):
    ''' 
    Filter data based on a list of Donor ID.  Merge all data together into one dataframe.  
    Arguments: list of 'Donor ID'
    Returns: dataframe 
    '''
    
    # To ensure ID parameter works for both a list and a string
    if isinstance(ID, list):
        temp = donations[donations['Donor ID'].isin(ID)].reset_index(drop = True)
    else:
        temp = donations[donations['Donor ID'] == ID].reset_index(drop = True)
    
    temp = temp.merge(donors, on = 'Donor ID', how='left')
    temp = temp.merge(projects, on = 'Project ID', how = 'left')
    temp = temp.merge(schools, on = 'School ID', how = 'left')
    
    return temp
def recommender(ID,
                donorDonations = None,
                donations_prior = donations_prior, 
                donors = donors, 
                projects = projects, 
                schools = schools, 
                donorFeatureMatrix = donorFeatureMatrix,
                projectFeatures = projectFeatures):
    '''
    Filter potential projects based on donor's previous donations, and donor's home location
    Rank projects based on dot product score of project features matrix, and donor feature matrix 
    
    Arguments: "Donor ID", 
                donations - the donation history (do not include future donations)
                donorFeatureMatrix - compiled using donor history to ensure no look ahead bias
    Returns: 
    Donor ID
    Ranking- Ranking of the project that the donor actually donated to 
    Project ID- Project ID of the project that the donor donated to
    Filter Size- # of projects that met location criterias
    Universe- dataframe of projectFeatures with scoring
    '''
    
    #print("Donor ID:", ID)
    
    ##
    # Merge donations data filtered by ID with donors, projects, schools
    ##
    
    if donorDonations is None:
        donorDonations = merge_data(ID, donations_prior)
    else:
        donorDonations = donorDonations.loc[donorDonations['Donor ID'] == ID]

    ##
    #  Get Previous School ID, Teacher ID, School Location (City/State), Donor's Location
    ##
    
    # previous locations of donations
    byLocation = donorDonations.groupby(['School City', 'School State'])['Donation Amount'].sum().sort_values(ascending = False)
    locMap = donorDonations.drop_duplicates(subset = ['School City', 'School State']).set_index((['School City', 'School State']))
    byLocation = byLocation.reset_index()
    
    # calculate the rank of city according to amount of donation received
    byLocation['byLocation rank']= byLocation['Donation Amount'].rank(ascending = False) # rank from largest to smallest
    
    # calculate the Frequent location bonus scoring scale
    byLocation['byLocation scale'] = 1/(byLocation['byLocation rank']/len(byLocation))/100
    
    # add longitude and latitude data for reference later
    byLocation = byLocation.set_index(['School City', 'School State'])
    byLocation = byLocation.merge(locMap.filter(items = ['School_Lon', 'School_Lat']), left_index = True, right_index = True, how = 'left')
    byLocation = byLocation.reset_index()
    byLocation['CityState'] = byLocation['School City'].map(str)+', '+byLocation['School State']
    
    # donor's location
    donorLocation = donors.loc[donors['Donor ID'] == ID, ['Donor City', 'Donor State', 'Donor_Lat', 'Donor_Lon']].reset_index(drop = True)
    donorLocation['CityState'] = donorLocation['Donor City'].map(str)+', '+donorLocation['Donor State']
    
    # get the ID masks for filtering donor donations
    schoolIDs = donorDonations['School ID'].unique()
    teacherIDs = donorDonations['Teacher ID'].unique()
    locationIDs = byLocation['CityState'].unique()
    homeIDs = donorLocation['CityState'].unique() 
    
    ##
    # Filter projectFeatures for Schools, Teachers, Locations that the donor has ever donated to
    ##
    
    projSchool = list(projectFeatures[projectFeatures['School ID'].isin(schoolIDs)]['Project ID'])
    projTeacher = list(projectFeatures[projectFeatures['Teacher ID'].isin(teacherIDs)]['Project ID'])
    projLocation = list(projectFeatures[projectFeatures['CityState'].isin(locationIDs)]['Project ID'])
    projHomeLoc = list(projectFeatures[projectFeatures['CityState'].isin(homeIDs)]['Project ID'])
    projAll = set(projSchool).union(set(projTeacher)).union(set(projLocation)).union(set(projHomeLoc))
    
    ##
    #  Filter the project Features based on 'Project IDs'
    ##
    
    projUniverse = projectFeatures.loc[projectFeatures['Project ID'].isin(projAll)]
    #print('Number of potential projects:', len(projUniverse))
    
    ##
    # Generate Donor Feature Matrix, Calculate Score, Rank Recommendations
    ##
    
    # Do the following if there are 1 or more potential projects to choose from
    if len(projAll) >=1: 
        
        #### Get Donor FeatureMatrix #### 
        
        donorFeatureMatrix = donorFeatureMatrix.set_index('Donor ID')  # if donorFeatureMatrix did not have index set
        y = donorFeatureMatrix.loc[ID, 'Project Type_Professional Development':'ProjCat_Warmth, Care & Hunger'] 
        y = y.values.reshape(len(y), 1) 

        #### Calculate Similarity Score
        score = np.dot(projUniverse.loc[:, 'Project Type_Professional Development':'ProjCat_Warmth, Care & Hunger'], y)

        #### Add more Scoring Attributes
        
        # score from dot product of similarity
        projUniverse['Score_interest'] = score
        
        # Flag as 1 if the project matches the conditions
        projUniverse['Score_School'] = projUniverse['Project ID'].isin(projSchool)
        projUniverse['Score_Teacher'] = projUniverse['Project ID'].isin(projTeacher)
        projUniverse['Score_homeLoc'] = projUniverse['Project ID'].isin(projHomeLoc)
        projUniverse['Score_priorLoc'] = projUniverse['Project ID'].isin(projLocation)
        
        # merge the scoring matrix
        projUniverse = projUniverse.merge(byLocation.filter(items = ['byLocation rank', 'byLocation scale', 'CityState']), on = 'CityState', how = 'left')

        #### Rank Recommendations #### 
        cols = ['Score_interest', 'Score_School', 'Score_Teacher','Score_homeLoc', 'Score_priorLoc']
        projUniverse['Score_Total_Unadjusted'] = projUniverse.loc[:, cols].sum(axis = 1)
        
        projUniverse['location_Premium'] = projUniverse['Score_Total_Unadjusted']*projUniverse['byLocation scale']

        cols = ['Score_interest', 'Score_School', 'Score_Teacher','Score_homeLoc', 'Score_priorLoc', 'location_Premium']
        projUniverse['Score_Total'] = projUniverse.loc[:, cols].sum(axis = 1)
        
        projUniverse['Rank'] = projUniverse['Score_Total'].rank(ascending = False)
        projUniverse = projUniverse.sort_values(by = 'Rank' )
        projUniverse = projUniverse.set_index('Project ID')
        
        # Length of potential project selections
        lenUniv = len(projUniverse)

    else:
        lenUniv = 0
    
    ##
    # Identify the actual project that the donor donated and find the recommender ranking of the correct project
    ##
    
    #### Get the actual project that the donor donated to
    ans = donations_2018[donations_2018['Donor ID'] == ID]['Project ID']
    
    # loop through multiple projects for donors that donated to multiple projects in the testing timeframe
    for i in range(len(ans)):
        proj = ans.values[i]

        ### Get ranking of the project that the donor donated to
        try:
            ranking = projUniverse.loc[proj]['Rank']
            
            # if project matches, skip the rest of the search
            break  
            
        except:
            # if cannot find matching project
            ranking = np.nan 
    #print('Ranking of correct response:', ranking)
        
    ### Return dictionary 
    response = { 'Donor ID': ID, 'Ranking': ranking, 'Donor Project ID': proj, 
                'Filter Size': lenUniv, 'Universe': projUniverse, 'Donor Donations': donorDonations,
               'Prior Location Count': len(locationIDs)}
    
    return response
response = recommender('01487813310e283992cfd5249c6cd722')
print("The project chosen by the Donor is ranked:", response['Ranking'])
print("The donor donated to Project ID:", response['Donor Project ID'])
print("Total projects in the filtered Universe is:", response['Filter Size'])
print("Number of projects the donor donated to prior to 2018:", len(response['Donor Donations']))
cols = ['Score_interest', 'Score_School', 'Score_Teacher', 'Score_homeLoc','Score_priorLoc', 'byLocation rank', 'byLocation scale','Score_Total_Unadjusted', 'location_Premium', 'Score_Total', 'Rank']
responseScores = response['Universe'][cols][0:5]
picks = response['Universe'].index[0:5]
recommendedProj = projects.loc[projects['Project ID'].isin(picks)]
TopRecommendations = recommendedProj.merge(schools, left_on= 'School ID', right_on = 'School ID', how = 'left')
TopRecommendations = TopRecommendations.merge(responseScores, left_on= 'Project ID', right_on = 'Project ID', how = 'left')
TopRecommendations.sort_values(by = 'Rank', ascending = True)
i = response['Ranking'].astype('int')
cols = ['Score_interest', 'Score_School', 'Score_Teacher', 'Score_homeLoc','Score_priorLoc', 'byLocation rank', 'byLocation scale','Score_Total_Unadjusted', 'location_Premium', 'Score_Total', 'Rank']
responseScores = response['Universe'][cols][i-1: i]
picks = response['Donor Project ID']
correctProj = projects.loc[projects['Project ID'] == picks]
correctProj = correctProj.merge(schools, left_on= 'School ID', right_on = 'School ID', how = 'left')
correctProj = correctProj.merge(responseScores, left_on= 'Project ID', right_on = 'Project ID', how = 'left')
correctProj 
response['Donor Donations'].sort_values(by = 'Donation Received Date', ascending = False).head()
donorSummary.loc[donorSummary['Donor ID'] == response['Donor ID']]
def accuracy(test_ID, 
             numSample, 
             donations_prior = donations_prior):
    
    donorDonations = merge_data(test_ID, donations_prior)
    recommendations = pd.DataFrame(columns=['Donor ID', 'Ranking', 'Donor Project ID', 
                                            'Filter Size', 'Prior Donation Count', 'Prior Location Count'])
    IDs = pd.Series(test_ID).sample(n= numSample, random_state = 513)
    recommendations['Donor ID'] = IDs
    recommendations = recommendations.set_index('Donor ID', drop = False)
    i = 1
    
    for ID in IDs:
        #print('Processing #:', i)
        response = recommender(ID, donorDonations)
        recommendations.loc[ID, 'Ranking'] = response['Ranking']
        recommendations.loc[ID, 'Donor Project ID'] = response['Donor Project ID']
        recommendations.loc[ID, 'Filter Size'] = response['Filter Size']
        recommendations.loc[ID, 'Prior Donation Count'] = len(response['Donor Donations'])
        recommendations.loc[ID, 'Prior Location Count'] = response['Prior Location Count']
        
        i+=1
    return recommendations
recommendations = accuracy(test_ID, 1000)
recommendations.head()
print('% of time the filter captured the correct project:', recommendations['Ranking'].notnull().sum()/len(recommendations))
chartData = recommendations.copy()
chartData['Ranking'] = chartData['Ranking'].astype('Float32')
chartData['Ranking Range'] = pd.cut(chartData['Ranking'], bins = [0, 1, 5, 10, 25, 50, 100, 70000], labels = ['1', '2-5', '6-10', '11-25', '26-50', '51-100', '>100'])
chartData['Ranking Range']= chartData['Ranking Range'].astype(str)
chartData = chartData.groupby('Ranking Range').agg('count')
chartData['Frequency'] = chartData['Donor ID']/chartData['Donor ID'].sum()*100
chartData = chartData.reindex(index = ['1', '2-5', '6-10', '11-25', '26-50', '51-100', '>100', 'nan'])

g =sns.barplot(chartData.index, chartData['Frequency'])
g.set(xlabel="Recommender's Ranking", ylabel='Frequency %', title = "Recommender's ranking of donor's chosen Project")

print('% Hit in top 1 Recommendations:', chartData['Frequency'][0:1].sum() )
print('% Hit in top 5 Recommendations:', chartData['Frequency'][0:2].sum() )
print('% Hit in top 10 Recommendations:', chartData['Frequency'][0:3].sum() )
print('% Hit in top 25 Recommendations:', chartData['Frequency'][0:4].sum() )
