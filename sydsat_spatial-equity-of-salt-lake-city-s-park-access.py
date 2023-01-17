from IPython.display import Image

SLC_parks_image = "/Users/Syd/Desktop/SLC_parks.png"

Image(SLC_parks_image, width = "500", height = "500")
SLC_census_image = "/Users/Syd/Desktop/SLC_census.png"

Image(SLC_census_image, width = '400', height = '400')
#warnings were turned off for aesthetic purposes.

import warnings

warnings.filterwarnings("ignore")
import pandas as pd #uses pandas to view, clean and tidy dataframes.



class Spatial_Park_Equity:

    '''This class creates a table of each census tract in Salt Lake City with its corresponding population,

    census tract area (m^2), park need (m^2), park area (m^2), and park supplied per capita.'''

    

    def __init__ (self, census_filepath, clipped_filepath, assumed_need = 10):

        #Initialize the path where the census dataframe is stored on the computer as a string. 

        self.census_filepath = census_filepath    #Ex: "/Users/Syd/Desktop/census_filename.xls."

        #Initialize the path where the clipped census dataframe is stored on the computer as a string.

        self.clipped_filepath = clipped_filepath  #Ex: "/Users/Syd/Desktop/clipped_filename.xls."

        #Initialize the assumed square meters of park needed per capita

        self.assumed_need = assumed_need

        

    def park_need(self):

        """This method opens the excel document that contains the census tract data. It tidies and 

        cleans the data using pandas and then calculates the park needed in each census tract and

        adds it as a new column. 

        """

        census_df = pd.read_excel(self.census_filepath, sep = "\t")       #imports excel doc

        #tidy data: subset variables

        census_need = census_df[['GEOID10','NAMELSAD10','DP0010001','Area']] #takes a subset of columns

        #clean data: rename column headers

        census_need = census_need.rename(columns = {'GEOID10' : 'geo_id'})        #rename to geo_id

        census_need = census_need.rename(columns = {'NAMELSAD10':'census_tract'}) #rename to census_tract

        census_need = census_need.rename(columns = {'DP0010001':'population'})    #rename to population

        census_need = census_need.rename(columns = {'AREA':'census_tract_area'})  #rename to census_tract_area

        #calculate and add new column: park_need

        census_need['park_need'] = (census_need['population']*self.assumed_need)

        #clean data: eliminate 0 or N/A data

        census_need = census_need.drop(31) #row 31 is not in census_area 

        census_need = census_need.drop(33) #row 33 has no data

        census_need = census_need.reset_index()

        census_need = census_need.drop(['index'], axis = 1)

        return census_need

    

    def park_supplied(self):

        """This method opens the excel document that contains the census tract park buffer data. It tidies and 

        cleans the data using pandas and then calculates the park supplied per person in each census tract and 

        adds it as a new column.

        """

        census_clip = pd.read_excel(self.clipped_filepath, sep = "\t")    #imports excel doc

        #tidy data: subset variables

        census_area = census_clip[['GEOID10','NAMELSAD10','DP0010001','AREA']] #takes a subset of columns

        #clean data: rename column headers

        census_area = census_area.rename(columns = {'GEOID10' : 'geo_id'})        #rename to geo_id

        census_area = census_area.rename(columns = {'NAMELSAD10':'census_tract'}) #rename to census_tract

        census_area = census_area.rename(columns = {'DP0010001':'population'})    #rename to population

        census_area = census_area.rename(columns = {'AREA':'park_area'})          #rename to park_area

        #calculate and add new column: park_supplied_per_cap

        census_area['park_supplied_per_cap'] = (census_area['park_area'] / census_area['population'])

        #clean data: eliminate 0 or N/a data

        census_area = census_area.drop(['geo_id','census_tract','population'], axis = 1)

        census_area = census_area.drop(32)

        census_area = census_area.reset_index()

        census_area = census_area.drop(['index'], axis = 1)

        return census_area

    

    def park_equity_dataframe(self):

        '''This method combines the park_need and park_supply tables into one table'''

        park_need = self.park_need()

        park_supplied = self.park_supplied()

        #combine park need and park supplied dataframes

        park_equity_dataframe = pd.concat([park_need, park_supplied], axis = 1)

        park_equity_dataframe['assumed_need'] = self.assumed_need

        return park_equity_dataframe

    

    def spatial_inequity_dataframe(self):

        '''This method determines if the amount of park supplied is spatially equitable or inequitable.

        Prints the number of inequitable census tracts and returns a table of inequitable census tracts'''

        equity_df = self.park_equity_dataframe()

        #Use boolean test to see if a census tract is spatially equitable or spatially inequitable

        equity_df['spatial_inequity'] = equity_df['park_need'] > equity_df['park_area']

        #Pull out spatially inequitable rows into new table: spatial_inequity

        spatial_inequity = equity_df[equity_df.spatial_inequity == True]

        spatial_inequity['assumed_need'] = self.assumed_need

        #Python treats True as 1 and False as 0, sum gives us number of Trues/inequitable tracts

        print('Number of spatially inequitable census tracts:', sum(equity_df['spatial_inequity']))

        return spatial_inequity

            

    def export_equity_dataframe(self, new_filename):

        '''This methods the finalized park equity table into an excel document. Takes one parameter,

        new_file which must be a string of the new filename with .xls at the end'''

        park_equity_export = self.park_equity_dataframe()

        park_equity_export.to_csv(new_filename)

        return 'exported!'
quarter_mile_10_assumed = Spatial_Park_Equity('/Users/Syd/Desktop/census_tract.xls', '/Users/Syd/Desktop/0.25_clip.xls', 10)
quarter_10 = quarter_mile_10_assumed.spatial_inequity_dataframe()
need_quarter_10 = quarter_mile_10_assumed.park_equity_dataframe()
quarter_mile_16_assumed = Spatial_Park_Equity('/Users/Syd/Desktop/census_tract.xls', '/Users/Syd/Desktop/0.25_clip.xls', 16)
quarter_16 = quarter_mile_16_assumed.spatial_inequity_dataframe()
need_quarter_16 = quarter_mile_16_assumed.park_equity_dataframe()
quarter_mile_20_assumed = Spatial_Park_Equity('/Users/Syd/Desktop/census_tract.xls', '/Users/Syd/Desktop/0.25_clip.xls', 20)
quarter_20 = quarter_mile_20_assumed.spatial_inequity_dataframe()
need_quarter_20 = quarter_mile_20_assumed.park_equity_dataframe()
third_mile_10_assumed = Spatial_Park_Equity('/Users/Syd/Desktop/census_tract.xls', '/Users/Syd/Desktop/0.33_clip.xls', 10)
third_10 = third_mile_10_assumed.spatial_inequity_dataframe()
need_third_10 = third_mile_10_assumed.park_equity_dataframe()
third_mile_16_assumed = Spatial_Park_Equity('/Users/Syd/Desktop/census_tract.xls', '/Users/Syd/Desktop/0.33_clip.xls', 16)
third_16 = third_mile_16_assumed.spatial_inequity_dataframe()
need_third_16 = third_mile_16_assumed.park_equity_dataframe()
third_mile_20_assumed = Spatial_Park_Equity('/Users/Syd/Desktop/census_tract.xls', '/Users/Syd/Desktop/0.33_clip.xls', 20)
third_20 = third_mile_20_assumed.spatial_inequity_dataframe()
need_third_20 = third_mile_20_assumed.park_equity_dataframe()
half_mile_10_assumed = Spatial_Park_Equity('/Users/Syd/Desktop/census_tract.xls', '/Users/Syd/Desktop/0.50_clip.xls', 10)
half_10 = half_mile_10_assumed.spatial_inequity_dataframe()
need_half_10 = half_mile_10_assumed.park_equity_dataframe()
half_mile_16_assumed = Spatial_Park_Equity('/Users/Syd/Desktop/census_tract.xls', '/Users/Syd/Desktop/0.50_clip.xls', 16)
half_16 = half_mile_16_assumed.spatial_inequity_dataframe()
need_half_16 = half_mile_16_assumed.park_equity_dataframe()
half_mile_20_assumed = Spatial_Park_Equity('/Users/Syd/Desktop/census_tract.xls', '/Users/Syd/Desktop/0.50_clip.xls', 20)
half_20 = half_mile_20_assumed.spatial_inequity_dataframe()
need_half_20 = half_mile_20_assumed.park_equity_dataframe()
three_quart_mile_10_assumed = Spatial_Park_Equity('/Users/Syd/Desktop/census_tract.xls', '/Users/Syd/Desktop/0.75_clip.xls', 10)
three_10 = three_quart_mile_10_assumed.spatial_inequity_dataframe()
three_quart_mile_16_assumed = Spatial_Park_Equity('/Users/Syd/Desktop/census_tract.xls', '/Users/Syd/Desktop/0.75_clip.xls', 16)
three_16 = three_quart_mile_16_assumed.spatial_inequity_dataframe()
three_quart_mile_20_assumed = Spatial_Park_Equity('/Users/Syd/Desktop/census_tract.xls', '/Users/Syd/Desktop/0.75_clip.xls', 20)
three_20 = three_quart_mile_20_assumed.spatial_inequity_dataframe()
one_mile_10_assumed = Spatial_Park_Equity('/Users/Syd/Desktop/census_tract.xls', '/Users/Syd/Desktop/1.00_clip.xls', 10)
one_10 = one_mile_10_assumed.spatial_inequity_dataframe()
one_mile_16_assumed = Spatial_Park_Equity('/Users/Syd/Desktop/census_tract.xls', '/Users/Syd/Desktop/1.00_clip.xls', 16)
one_16 = one_mile_16_assumed.spatial_inequity_dataframe()
one_mile_20_assumed = Spatial_Park_Equity('/Users/Syd/Desktop/census_tract.xls', '/Users/Syd/Desktop/1.00_clip.xls', 20)
one_20 = one_mile_20_assumed.spatial_inequity_dataframe()
half_16
quarter_mile = pd.concat([quarter_10, quarter_16, quarter_20])

quarter_mile['walking_distance'] = 0.25

quarter_mile = quarter_mile.reset_index()

quarter_mile = quarter_mile.drop(['index'], axis = 1)

#quarter_mile.head()
third_mile = pd.concat([third_10, third_16, third_20])

third_mile['walking_distance'] = 0.33

third_mile = third_mile.reset_index()

third_mile = third_mile.drop(['index'], axis = 1)

#third_mile.head()
half_mile = pd.concat([half_10, half_16, half_20])

half_mile['walking_distance'] = 0.50

half_mile = half_mile.reset_index()

half_mile = half_mile.drop(['index'], axis = 1)

#half_mile.head()
inequity_combine = pd.concat([quarter_mile, third_mile, half_mile])

#inequity_combine
#make new dataframe that only includes walking distance and spatial inequity

walking_total = inequity_combine[['walking_distance','spatial_inequity']]



#create new dataframe for 0.25 mile walking distance

quarter_summarized = walking_total[walking_total.walking_distance == 0.25]

#make new row, inequity_sum, that is the number of inequitable census tracts

quarter_summarized['inequity_sum'] = sum(quarter_summarized['spatial_inequity'])

#simplify data to just first row

quarter_summarized = quarter_summarized.head(1)



#create new dataframe for 0.33 mile walking distance

third_summarized = walking_total[walking_total.walking_distance == 0.33]

third_summarized['inequity_sum'] = sum(third_summarized['spatial_inequity'])

third_summarized = third_summarized.head(1)



#create new dataframe for 0.75 mile walking distance

half_summarized = walking_total[walking_total.walking_distance == 0.50]

half_summarized['inequity_sum'] = sum(half_summarized['spatial_inequity'])

half_summarized = half_summarized.head(1)



#put all three walking distances into one dataframe

walking_summarized = pd.concat([quarter_summarized, third_summarized, half_summarized])

#cleaning walking_summarized: sort by walking distance, remove 'spatial_inequity', reset index

walking_summarized = walking_summarized.sort_values('walking_distance')

walking_summarized = walking_summarized[['walking_distance','inequity_sum']]

walking_summarized = walking_summarized.reset_index()

walking_summarized = walking_summarized.drop(['index'], axis = 1)
#make new dataframe that only includes assumed and spatial inequity

need_total = inequity_combine[['assumed_need','spatial_inequity']]



#create new dataframe for 10 m^2 assumed need

ten_summarized = need_total[need_total.assumed_need == 10]

#make new row, inequity_sum, that is the number of inequitable census tracts

ten_summarized['inequity_sum'] = sum(ten_summarized['spatial_inequity'])

#simplify data to just first row

ten_summarized = ten_summarized.head(1)



#create new dataframe for 16 m^2 assumed need

sixteen_summarized = need_total[need_total.assumed_need == 16]

sixteen_summarized['inequity_sum'] = sum(sixteen_summarized['spatial_inequity'])

sixteen_summarized = sixteen_summarized.head(1)



#create new dataframe for 20 m^2 assumed need

twenty_summarized = need_total[need_total.assumed_need == 20]

twenty_summarized['inequity_sum'] = sum(twenty_summarized['spatial_inequity'])

twenty_summarized = twenty_summarized.head(1)



#put all three assumed needs into one dataframe

need_summarized = pd.concat([ten_summarized, sixteen_summarized, twenty_summarized])



#cleaning walking_summarized: sort by walking distance, remove 'spatial_inequity', reset index

need_summarized = need_summarized.sort_values('assumed_need')

need_summarized = need_summarized[['assumed_need','inequity_sum']]

need_summarized = need_summarized.reset_index()

need_summarized = need_summarized.drop(['index'], axis = 1)
def plot_walking():

    #Import matplotlib, this is the package used to graph.

    import matplotlib.pyplot as plt

    #pulling data from walking_summarized dataframe.

    plt.plot(walking_summarized['walking_distance'], walking_summarized['inequity_sum'], color='b')

    #set x-axis range

    plt.xlim([0.25,0.50])

    #labelling plot

    plt.xlabel('Walking Distance (miles)')

    plt.ylabel('Number of Spatially Inequitable Census Tracts')

    plt.title('Access to Parks Across Varying Walking Distance Assumptions')

    plt.show()
def plot_need():

    #Import matplotlib, this is the package used to graph.

    import matplotlib.pyplot as plt

    #pulling data from walking_summarized dataframe.

    plt.bar(need_summarized['assumed_need'], need_summarized['inequity_sum'], color='b', align = 'center')



    #labelling plot

    plt.xlabel('Assumed Need (meters squared)')

    plt.ylabel('Number of Spatially Inequitable Census Tracts')

    plt.title('Access to Parks Across Varying Park Need Assumptions')

    plt.show()
walking_summarized
plot_walking()
need_summarized
plot_need()
from IPython.display import Image

quarter_mile_image = "/Users/Syd/Desktop/quarter_mile.jpg"

Image(quarter_mile_image, width = "1000", height = "1000")
from IPython.display import Image

third_mile_image = "/Users/Syd/Desktop/third_mile.jpg"

Image(third_mile_image, width = "1000", height = "1000")
from IPython.display import Image

half_mile_image = "/Users/Syd/Desktop/half_mile.jpg"

Image(half_mile_image, width = "1000", height = "1000")
#pull together the geo_ids for the salt lake city to be able to compare to income data

census_tracts = half_mile_10_assumed.park_need()

census_tracts = census_tracts.sort_values('geo_id')

census_tracts = census_tracts.reset_index()

census_tracts = census_tracts.drop(['index'], axis = 1)

#census_tracts.tail()
#import income dataframe

income_df = pd.read_excel("/Users/Syd/Desktop/census_income.xls", sep = "\t") 

#pull out only 'GEOID' and 'B19013e1' columns

income_df = income_df[['GEOID','B19013e1']]

#rename columns

income_df = income_df.rename(columns = {'GEOID':'geo_id1', 'B19013e1': 'median_income'})

#drop n/a data

income_df = income_df.dropna()

#change float to int for median income

income_df['median_income'] = income_df['median_income'].astype(int)

#remove '14000US' prefix from geo_id

income_df['geo_id1'].replace("14000US", "", regex = True, inplace = True)

income_df['geo_id1'] = income_df['geo_id1'].astype(int)

#filter the census tracts that are within salt lake city

income_df = income_df[income_df.geo_id1.isin(census_tracts.geo_id)]

#reset index

income_df = income_df.reset_index()

income_df = income_df.drop(['index'], axis = 1)



combined_df = pd.concat([census_tracts, income_df], axis=1)

#drop geo_id1 column

combined_df = combined_df[['geo_id', 'census_tract', 'population', 'Area', 'park_need', 'median_income']]

combined_df = combined_df.rename(columns = {'Area':'park_supplied'})

income_df = income_df.rename(columns = {'geo_id1':'geo_id'})

#combined_df
#add walking distances

need_quarter_10['walking_distance'] = 0.25

need_quarter_16['walking_distance'] = 0.25

need_quarter_20['walking_distance'] = 0.25

need_third_10['walking_distance'] = 0.33

need_third_16['walking_distance'] = 0.33

need_third_20['walking_distance'] = 0.33

need_half_10['walking_distance'] = 0.50

need_half_16['walking_distance'] = 0.50

need_half_20['walking_distance'] = 0.50



spatial_combined = pd.concat([need_quarter_10, need_quarter_16, need_quarter_20, need_third_10, need_third_16, need_third_20, need_half_10, need_half_16, need_half_20])

spatial_combined = spatial_combined.reset_index()

spatial_combined = spatial_combined.drop(['index'], axis = 1)

#add median_income to all spatial data

spatial_combined = pd.merge(income_df, spatial_combined, how='right', on= 'geo_id')

spatial_combined['spatial_inequity'] = spatial_combined['park_need'] > spatial_combined['park_area']

#spatial_combined

spatial_inequitable_income = spatial_combined[spatial_combined.spatial_inequity == True]

spatial_equitable_income = spatial_combined[spatial_combined.spatial_inequity == False]
spatial_inequitable_income = spatial_inequitable_income.reset_index()

spatial_inequitable_income = spatial_inequitable_income.drop(['index'], axis = 1)

#spatial_inequitable_income.head()
spatial_equitable_income = spatial_equitable_income.reset_index()

spatial_equitable_income = spatial_equitable_income.drop(['index'], axis = 1)

#spatial_equitable_income.head()
def plot_income_scatter():

    import seaborn as sns

    import matplotlib.pyplot as plt



    ## shows graph in notebook

    %matplotlib inline



    ## makes a figure object

    plt.figure()

    ## gets rid of dark grey background

    sns.set(style='whitegrid', color_codes=True) # change 'whitegrid' to 'white' and grid goes away

    ## plots our data pulling columns from dataframe

    sns.stripplot(x='spatial_inequity', y='median_income', data=spatial_combined, jitter=True) # jitter for overplotting

    ## gets rid of left and right "box lines"

    sns.despine(left=True)

    plt.gca().set_title('Spatial Equity and Median Income')

    plt.gca().set_ylabel('Median Income per household (2015 U.S. Inflated Dollars)')

    plt.gca().set_xlabel('Spatial Equity/Inequity (number of census tracts)');
def plot_income_box():

    import seaborn as sns

    import matplotlib.pyplot as plt



    ## shows graph in notebook

    %matplotlib inline



    ## makes a figure object

    plt.figure()

    ## gets rid of dark grey background

    sns.set(style='whitegrid', color_codes=True) # change 'whitegrid' to 'white' and grid goes away

    ## plots our data pulling columns from dataframe

    sns.boxplot(x="spatial_inequity", y="median_income", data=spatial_combined)

    

    ## gets rid of left and right "box lines"

    sns.despine(offset=10, trim=True)

    plt.gca().set_title('Spatial Equity and Median Income')

    plt.gca().set_ylabel('Median Income per household (2015 U.S. Inflated Dollars)')

    plt.gca().set_xlabel('Spatial Equity/Inequity (number of census tracts)');
def plot_income_box_walking():

    import seaborn as sns

    import matplotlib.pyplot as plt



    ## shows graph in notebook

    %matplotlib inline



    ## makes a figure object

    plt.figure()

    ## gets rid of dark grey background

    sns.set(style='whitegrid', color_codes=True) # change 'whitegrid' to 'white' and grid goes away

    ## plots our data pulling columns from dataframe

    sns.boxplot(x="spatial_inequity", y="median_income", hue="walking_distance", data=spatial_combined)

    

    ## gets rid of left and right "box lines"

    sns.despine(offset=10, trim=True)

    plt.gca().set_title('Spatial Equity and Median Income')

    plt.gca().set_ylabel('Median Income per household (2015 U.S. Inflated Dollars)')

    plt.gca().set_xlabel('Spatial Equity/Inequity (number of census tracts)');
def plot_income_box_need():

    import seaborn as sns

    import matplotlib.pyplot as plt



    ## shows graph in notebook

    %matplotlib inline



    ## makes a figure object

    plt.figure()

    ## gets rid of dark grey background

    sns.set(style='whitegrid', color_codes=True) # change 'whitegrid' to 'white' and grid goes away

    ## plots our data pulling columns from dataframe

    sns.boxplot(x="spatial_inequity", y="median_income", hue="assumed_need", data=spatial_combined)

    

    ## gets rid of left and right "box lines"

    sns.despine(offset=10, trim=True)

    plt.gca().set_title('Spatial Equity and Median Income')

    plt.gca().set_ylabel('Median Income per household (2015 U.S. Inflated Dollars)')

    plt.gca().set_xlabel('Spatial Equity/Inequity (number of census tracts)');
plot_income_scatter()
plot_income_box()
plot_income_box_walking()
plot_income_box_need()
from IPython.display import Image

income_image = "/Users/Syd/Desktop/income_level.jpg"

Image(income_image, width = "1000", height = "1000")