from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

print(os.listdir('../input'))

# Read in the DataFrame:

nRowsRead = None # specify 'None' if want to read whole file

df1 = pd.read_csv('../input/EPS_Neighbourhood_Criminal_Incidents.csv', delimiter=',', nrows = nRowsRead)



# Change Column Names for Easier Use:

df1.columns = ['Neighbourhood_Name', 'Violation_Type', 'Year', 'Quarter', 'Month', 'Number_Incidents']



# Generate a Base Dataset For the rest of the function to use and reset the index, as we will adjust the index in our graphing function:

df2 = pd.DataFrame(df1.groupby(['Year', 'Neighbourhood_Name', 'Violation_Type']).Number_Incidents.agg(sum))

df2 = df2.reset_index()

df2.head()

def incidents_by_neighbourhood(neighbourhood_name):

    """Function to create a dataframe isolating a neighbourhood name, 

    Returning a dataframe displaying the Sum of Incidents, for a Specific Violation Type in a Specific Year

    Index: NONE

    PURPOSE: To Generate a Neighbourhood Dataframe that can be manipulated, graphed etc.

    neighbourhood_name: enter the name of the neighbourhood you wish to generate"""

    

    # Generate a DataFrame for the specified Neighbourhood_Name

    df_neighbourhood = pd.DataFrame(df2.loc[df2['Neighbourhood_Name'] == str(neighbourhood_name).upper()].groupby(['Year', 'Violation_Type']).Number_Incidents.agg(sum))

    

    # Reset Index

    df_neighbourhood = df_neighbourhood.reset_index()

    

    # A series of all the unique years included in our dataset. This will be our index or row labels.

    incident_year = pd.Series.unique(df2['Year'])

    

    # A series of all the unique Violation types. This will be our column labels in the final dataset. 

    violation_type = pd.Series.unique(df2['Violation_Type'])

    

    # An Empty DataFrame with Year as Index, and the various Incident_Types as columns

    df_final = pd.DataFrame(columns=violation_type, index=incident_year)

    

    # Writes into the Empty Dataframe the sum of each incident type by year using the specific neighbourhood data obtained in df_neighbourhood

    for i in incident_year:

        df_final.loc[i] = pd.Series(df_neighbourhood.groupby(['Year', 'Violation_Type']).Number_Incidents.sum()[i])

        

    print(neighbourhood_name.upper() + " Incidents By Year/Type")

    

    # Generate the Graphs of the Crimes by Type for the Neighbourhood

    for t in violation_type:

        df_final[t].plot.line(

            figsize=(10, 5),

            fontsize=16

        )

        plt.title(t.upper() + " INCIDENTS " + neighbourhood_name.upper() + " (Edmonton, AB)", fontsize=15)

        plt.xlabel('YEAR', fontsize=15)

        plt.ylabel('# of INCIDENTS', fontsize=15)

        plt.show()

# Type in a neighbourhood name from Edmonton and see if it works



# To try it yourself, get a map of Edmonton AB on google, zoom in to see the neighbourhood name 



# You could have this be a input() function which requests a neighbourhood from the user and displays the graph but since this is unsupported,

# if you want to check out another neighbourhood just fork this kernel and change the name in the string neighbourhood_to_graph below. 



neighbourhood_to_graph = 'downtown'



incidents_by_neighbourhood(neighbourhood_to_graph)
