import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.preprocessing import StandardScaler
def remove_percent_sign(string):
    """ Removing percent sign"""
    if pd.isnull(string):
        return string
    else:
        return string.replace('%', '')

def adding_proportions(df, dividend, divider,new_col_names):
    """Dividing columns in order to get proportion data"""
    i= 0
    for i in range(len(dividend)):
        df[new_col_names[i]] = df[dividend[i]]/df[divider[i]]
        i+=1

def get_reg_data(df, X_column_name, y_column_name, remove_zeros=False):
    """ Preparing columns for linear regression """
    df_name=df
    df_name[X_column_name]=pd.to_numeric(df_name[X_column_name], errors ='ignore')
    df_name[y_column_name]=pd.to_numeric(df_name[y_column_name], errors ='ignore')
   
    
    if remove_zeros:
        df_name= df_name[df_name[X_column_name]!=0]
        df_name= df_name[df_name[y_column_name]!=0]
    
    data_for_reg = df_name[[X_column_name, y_column_name]].dropna()
    
    x = data_for_reg[X_column_name]
    y = data_for_reg[y_column_name]
    
    return x, y

def generate_reg_graph(df, X_column, y_column, no_zeros=False):
    """ Peforms linear regression and graphs, can be provided with a list of columns"""
    if type(X_column) != list:
        X_column =[X_column]   
    if type(y_column) !=list:
        y_column = [y_column]
    for x_col in X_column:
            for y_col in y_column:
                x, y = get_reg_data(df, x_col, y_col,remove_zeros=no_zeros )
                if len(x) == len(y):
                    n=len(x)
                

                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                plt.scatter( x, y)
                if p_value<0.05:
                    plt.plot(x, intercept + slope * x, '-', color='red')
                
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.show()
                print('x:{}, y:{}, r_value:{}, p_value:{:.3f}, n :{}'.format(x_col, y_col, r_value, p_value, n))
              
#Reading, cleaning and adding calculations for the school data
school_data = pd.read_csv('../input/data-science-for-good/2016 School Explorer.csv')
str_cols_bool = (school_data.dtypes == object)
str_cols = list(str_cols_bool[str_cols_bool].index)

for col in str_cols:
    school_data[col]=school_data[col].apply(remove_percent_sign)    

ela_4s = ['Grade 3 ELA 4s - All Students',
          'Grade 4 ELA 4s - All Students',
          'Grade 5 ELA 4s - All Students',
          'Grade 6 ELA 4s - All Students',
          'Grade 7 ELA 4s - All Students',
          'Grade 8 ELA 4s - All Students']

ela_tested =['Grade 3 ELA - All Students Tested',
             'Grade 4 ELA - All Students Tested',
             'Grade 5 ELA - All Students Tested',
             'Grade 6 ELA - All Students Tested',
             'Grade 7 ELA - All Students Tested',
             'Grade 8 ELA - All Students Tested']

math_4s = ['Grade 3 Math 4s - All Students',
          'Grade 4 Math 4s - All Students',
          'Grade 5 Math 4s - All Students',
          'Grade 6 Math 4s - All Students',
          'Grade 7 Math 4s - All Students',
          'Grade 8 Math 4s - All Students']

math_tested= ['Grade 3 Math - All Students tested',
             'Grade 4 Math - All Students Tested',
             'Grade 5 Math - All Students Tested',
             'Grade 6 Math - All Students Tested',
             'Grade 7 Math - All Students Tested',
             'Grade 8 Math - All Students Tested']

ela_new_cols = ['Grade 3 ELA 4s proportion',
          'Grade 4 ELA 4s proportion',
          'Grade 5 ELA 4s proportion',
          'Grade 6 ELA 4s proportion',
          'Grade 7 ELA 4s proportion',
          'Grade 8 ELA 4s proportion']

math_new_cols =['Grade 3 Math 4s proportion',
          'Grade 4 Math 4s proportion',
          'Grade 5 Math 4s proportion',
          'Grade 6 Math 4s proportion',
          'Grade 7 Math 4s proportion',
          'Grade 8 Math 4s proportion']

adding_proportions(school_data, ela_4s, ela_tested, ela_new_cols)
adding_proportions(school_data, math_4s, math_tested, math_new_cols)
#Reading and adding calculations for SHSAT data
shsat_data = pd.read_csv('../input/data-science-for-good/D5 SHSAT Registrations and Testers.csv')
shsat_data['Rate of students taking SHSAT']=shsat_data['Number of students who took the SHSAT']/shsat_data['Enrollment on 10/31']
shsat_data['registered/enrolled'] = shsat_data['Number of students who registered for the SHSAT']/shsat_data['Enrollment on 10/31']
shsat_data['took/registered'] = shsat_data['Number of students who took the SHSAT']/shsat_data['Number of students who registered for the SHSAT']
avg_shsat = shsat_data.groupby('School name')['Rate of students taking SHSAT', 'registered/enrolled', 'took/registered'].mean()
avg_shsat = avg_shsat.reset_index()
avg_shsat['School Name']=avg_shsat['School name'].apply(lambda x: x.upper())
schools_shsat = school_data.merge(avg_shsat, how='inner')
schools_shsat['Student Attendance Rate'].replace(0,np.nan, inplace=True)

plt.figure(figsize=(16,16))
ax1 = plt.subplot(3,2,1)
sns.swarmplot(x= 'Strong Family-Community Ties Rating',  y='Rate of students taking SHSAT' , data = schools_shsat)
plt.ylabel('Rate of students taking the SHSAT')

plt.subplot(3,2,2, sharex=ax1)
sns.swarmplot(x= 'Rigorous Instruction Rating',  y='Rate of students taking SHSAT' , data = schools_shsat,
              order = ['Exceeding Target','Meeting Target', 'Approaching Target'] )
plt.ylabel('Rate of students taking the SHSAT')
plt.subplot(3,2,3, sharex=ax1)
sns.swarmplot(x= 'Collaborative Teachers Rating',  y='Rate of students taking SHSAT' , data = schools_shsat, 
              order = ['Exceeding Target','Meeting Target', 'Approaching Target'] )
plt.ylabel('Rate of students taking the SHSAT')
plt.subplot(3,2,4,sharex=ax1)
sns.swarmplot(x= 'Effective School Leadership Rating',  y='Rate of students taking SHSAT' , data = schools_shsat,
             order = ['Exceeding Target','Meeting Target', 'Approaching Target'] )
plt.ylabel('Rate of students taking the SHSAT')
plt.subplot(3,2,5,sharex=ax1)
sns.swarmplot(x= 'Trust Rating',  y='Rate of students taking SHSAT' , data = schools_shsat,
             order = ['Exceeding Target','Meeting Target', 'Approaching Target'] )
plt.ylabel('Rate of students taking the SHSAT')

plt.subplot(3,2,6,sharex=ax1)
sns.swarmplot(x= 'Student Achievement Rating',  y='Rate of students taking SHSAT' , data = schools_shsat,
             order = ['Exceeding Target','Meeting Target', 'Approaching Target'] )
plt.ylabel('Rate of students taking the SHSAT')
plt.show()
def perform_anova(df, category,col_of_interest ):
    exceeds = df[df[category]=='Exceeding Target'].loc[:,col_of_interest ]
    meets = df[df[category]=='Meeting Target'].loc[:,col_of_interest ]
    approaches = df[df[category]=='Approaching Target'].loc[:,col_of_interest ]
    F_value, p_value = stats.f_oneway(exceeds, meets, approaches)
    print('{}, {}, F_value = {:.3f}, p_value = {:.3f}\n'.format(category, col_of_interest,F_value, p_value))
perform_anova(schools_shsat, 'Effective School Leadership Rating', 'Rate of students taking SHSAT' )
perform_anova(schools_shsat, 'Student Achievement Rating', 'Rate of students taking SHSAT' )
cols_of_interest = ['Grade 8 ELA 4s proportion',
                    'Percent ELL', 
                   'Percent Black / Hispanic',
                   'Percent Black',
                   'Percent Hispanic',
                   'Percent Asian',
                   'Percent White',
                   'Average ELA Proficiency',
                    'Average Math Proficiency',
 'Grade 6 ELA 4s proportion',
 'Grade 7 ELA 4s proportion',
 'Grade 8 ELA 4s proportion',
 'Grade 6 Math 4s proportion',
 'Grade 7 Math 4s proportion',
 'Grade 8 Math 4s proportion']
                    
generate_reg_graph(schools_shsat, X_column='Student Attendance Rate', y_column= 'Rate of students taking SHSAT', no_zeros=True)
generate_reg_graph(schools_shsat, X_column=cols_of_interest, y_column= 'Rate of students taking SHSAT', no_zeros=False)

schma = pd.read_csv('../input/schema/schma19962016.csv', low_memory=False)

diploma_rate = schma[['SCHNAM', 'YEAR', 'HPGPERDIP4AVG']].dropna()

diploma_rate.rename(columns={'SCHNAM': 'School Name', 'YEAR': 'Year','HPGPERDIP4AVG':'Diploma Rate'}, inplace=True)

#diploma_rate['Year'].unique()

diploma_rate_2010_2013= diploma_rate[diploma_rate['Year']>2009]

diploma_rate_2010_2013= diploma_rate_2010_2013.dropna()

diploma_rate_2010_2013= diploma_rate_2010_2013.groupby('School Name')['Diploma Rate'].mean().to_frame()

diploma_rate_2010_2013=diploma_rate_2010_2013.reset_index()
schools_diploma_rate = diploma_rate_2010_2013.merge(school_data)
cols_to_check = ['Economic Need Index',
                 'Percent ELL',
 'Percent Black / Hispanic',
 'Percent Black',
 'Percent Hispanic',
 'Percent Asian',
 'Percent White',
 'Average ELA Proficiency',
 'Average Math Proficiency']
generate_reg_graph(schools_diploma_rate, X_column='Student Attendance Rate', y_column= 'Diploma Rate', no_zeros=True)
generate_reg_graph(schools_diploma_rate, X_column=cols_to_check, y_column= 'Diploma Rate', no_zeros=False)
plt.figure(figsize=(16,16))
ax1 = plt.subplot(3,2,1)
sns.violinplot(x= 'Strong Family-Community Ties Rating',  y='Average ELA Proficiency' , data = school_data)
#plt.ylabel('Rate of students taking the SHSAT')

plt.subplot(3,2,2, sharex=ax1)
sns.violinplot(x= 'Collaborative Teachers Rating',  y='Average ELA Proficiency' , data = school_data,
              order = ['Exceeding Target','Meeting Target', 'Approaching Target'] )
#plt.ylabel('Rate of students taking the SHSAT')
plt.subplot(3,2,3, sharex=ax1)
sns.violinplot(x= 'Rigorous Instruction Rating',  y='Average ELA Proficiency' , data = school_data, 
              order = ['Exceeding Target','Meeting Target', 'Approaching Target'] )
#plt.ylabel('Rate of students taking the SHSAT')
plt.subplot(3,2,4,sharex=ax1)
sns.violinplot(x= 'Effective School Leadership Rating',y='Average ELA Proficiency' , data = school_data,
             order = ['Exceeding Target','Meeting Target', 'Approaching Target'] )
#plt.ylabel('Rate of students taking the SHSAT')
plt.subplot(3,2,5,sharex=ax1)
sns.violinplot(x= 'Trust Rating',y='Average ELA Proficiency' , data = school_data,
             order = ['Exceeding Target','Meeting Target', 'Approaching Target'] )
#plt.ylabel('Rate of students taking the SHSAT')

plt.subplot(3,2,6,sharex=ax1)
sns.violinplot(x= 'Student Achievement Rating',y='Average ELA Proficiency' , data = school_data,
             order = ['Exceeding Target','Meeting Target', 'Approaching Target'] )
plt.ylabel('Rate of students taking the SHSAT')
plt.show()
perform_anova(school_data, 'Strong Family-Community Ties Rating', 'Average ELA Proficiency' )
perform_anova(school_data, 'Rigorous Instruction Rating', 'Average ELA Proficiency' )
perform_anova(school_data, 'Effective School Leadership Rating', 'Average ELA Proficiency' )
perform_anova(school_data, 'Collaborative Teachers Rating', 'Average ELA Proficiency' )
perform_anova(school_data, 'Trust Rating', 'Average ELA Proficiency' )
perform_anova(school_data, 'Student Achievement Rating', 'Average ELA Proficiency' )
cols_for_ela = ['Economic Need Index',
                'Percent ELL', 
                'Percent Black / Hispanic',
                'Percent Black',
                'Percent Hispanic',
                'Percent Asian',
                'Percent White']
                
generate_reg_graph(school_data, X_column='Student Attendance Rate', y_column= 'Average ELA Proficiency', no_zeros=True)
generate_reg_graph(school_data, X_column=cols_for_ela, y_column= 'Average ELA Proficiency', no_zeros=False)
cols_demographics = ['Percent ELL', 
                'Percent Black / Hispanic',
                'Percent Black',
                'Percent Hispanic',
                'Percent Asian',
                'Percent White']
generate_reg_graph(school_data, X_column=cols_demographics, y_column= 'Economic Need Index', no_zeros=False)
schools_need = school_data.loc[:, ['School Name','Economic Need Index', 'Student Attendance Rate']].sort_values(by=['Economic Need Index', 'Student Attendance Rate'], ascending = [False, True])
the_ten_in_need = schools_need.iloc[0:9,:]
the_ten_in_need
