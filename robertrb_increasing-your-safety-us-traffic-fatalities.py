# Imports of libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from google.cloud import bigquery as bq
from bq_helper import BigQueryHelper
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from io import StringIO
init_notebook_mode(connected=True)
# Useful functions
def converted_query_size(helper,query=''):
    """
    returns a human format for query size.
    Author: Robert Barbosa"""
    all_sizes=['GB','MB','KB','B']
    if type(helper) == BigQueryHelper:
        size =helper.estimate_query_size(query)
    i=0
    factor=1024
    while size<1:
        size*=factor
        i+=1
    return f'{size:.1f} {all_sizes[i]}'

def describe_tables(helper,table=None,only_name=False):
    width=104
    def print_table(t):
        print(f'Table: {t}\n{"-"*100}')
        for f in helper.table_schema(t):
            R = width - len(t) - len(f.name) - len(f.field_type)
            print(f'{f.name}: {f.description[:R]}{"..." if len(f.description)>R else ""} [{f.field_type}]')
    if table == None:
        for t in helper.list_tables():
            print_table(t)
    else:
        if only_name:
            for f in helper.table_schema(table):
                print(f.name)
        else:
            print_table(table)

def get_tablefield_description(helper,table,field):
    for f in helper.table_schema(table):
        if field == f.name:
            return f.description
# Project only functions
def plot_on_usa_map(df,loc,data_column,bar_title,layout_title,text=""):
    scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
                [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

    df['text'] = text

    data = [ dict(
            type='choropleth',
            colorscale = scl,
            autocolorscale = False,
            locations = loc,
            z = df[data_column],
            locationmode = 'USA-states',
            text = df['text'],
            marker = dict( line = dict ( color = 'rgb(255,255,255)', width = 2  ) ),
            colorbar = dict( title = bar_title)
            ) ]

    layout = dict(
            title = layout_title,
            geo = dict(
                scope='usa',
                projection=dict( type='albers usa' ),
                showlakes = True,
                lakecolor = 'rgb(255, 255, 255)'),
                 )

    fig = dict( data=data, layout=layout )
    iplot( fig, filename='d3-cloropleth-map' )
bq_assistant = BigQueryHelper('bigquery-public-data', 'nhtsa_traffic_fatalities')
describe_tables(bq_assistant)
describe_tables(bq_assistant,'accident_2016',True)
print(get_tablefield_description(bq_assistant,'accident_2016','first_harmful_event_name'))
bq_assistant.table_schema('accident_2016')
bq_assistant.head('accident_2016',5)
QUERY = """
        SELECT *
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
        """
print(f'Query size: {converted_query_size(bq_assistant,QUERY)}')
accident_2016 = bq_assistant.query_to_pandas_safe(QUERY)
accident_2016.info()
# Drop some irrelevant information
non_relevant_cols = [accident_2016.columns[i] for i in range(accident_2016.shape[1]) 
                     if accident_2016.dtypes[i] not in ['int64','float64']]

filter_cols = ['number_of_fatalities','number_of_drunk_drivers']
filter_cols.extend(['number_of_motor_vehicles_in_transport_mvit','number_of_parked_working_vehicles',
                    'number_of_persons_not_in_motor_vehicles_in_transport_mvit',
                    'number_of_persons_in_motor_vehicles_in_transport_mvit'])
accident_2016[filter_cols].describe().transpose().drop(['count','std','50%'],axis=1)
# Data from Wikipedia (Population on july 2017)
usa_population="""
state_name,population
Alabama,4874747
Alaska,739795
Arizona,7016270
Arkansas,3004279
California,39536653
Colorado,5607154
Connecticut,3588184
Delaware,961939
District of Columbia,693972
Florida,20984400
Georgia,10429379
Hawaii,1427538
Idaho,1716943
Illinois,12802023
Indiana,6666818
Iowa,3145711
Kansas,2913123
Kentucky,4454189
Louisiana,4684333
Maine,1335907
Maryland,6052177
Massachusetts,6859819
Michigan,9962311
Minnesota,5576606
Mississippi,2984100
Missouri,6113532
Montana,1050493
Nebraska,1920076
Nevada,2998039
New Hampshire,1342795
New Jersey,9005644
New Mexico,2088070
New York,19849399
North Carolina,10273419
North Dakota,755393
Ohio,11658609
Oklahoma,3930864
Oregon,4142776
Pennsylvania,12805537
Rhode Island,1059639
South Carolina,5024369
South Dakota,869666
Tennessee,6715984
Texas,28304596
Utah,3101833
Vermont,623657
Virginia,8470020
Washington,7405743
West Virginia,1815857
Wisconsin,5795483
Wyoming,579315
"""
df_usa_population = pd.read_csv(StringIO(usa_population))
df_usa_population.info()
#if "population" not in accident_2016.columns:
#    accident_2016 = accident_2016.merge(df_usa_population,how="outer",on="state_name")
#    print('population inserted on df')
# Used to plot on map
usa_states="""
code,state
AL,Alabama
AK,Alaska
AZ,Arizona
AR,Arkansas
CA,California
CO,Colorado
CT,Connecticut
DE,Delaware
FL,Florida
GA,Georgia
HI,Hawaii
ID,Idaho
IL,Illinois
IN,Indiana
IA,Iowa
KS,Kansas
KY,Kentucky
LA,Louisiana
ME,Maine
MD,Maryland
MA,Massachusetts
MI,Michigan
MN,Minnesota
MS,Mississippi
MO,Missouri
MT,Montana
NE,Nebraska
NV,Nevada
NH,New Hampshire
NJ,New Jersey
NM,New Mexico
NY,New York
NC,North Carolina
ND,North Dakota
OH,Ohio
OK,Oklahoma
OR,Oregon
PA,Pennsylvania
RI,Rhode Island
SC,South Carolina
SD,South Dakota
TN,Tennessee
TX,Texas
UT,Utah
VT,Vermont
VA,Virginia
WA,Washington
WV,West Virginia
WI,Wisconsin
WY,Wyoming
"""
df_usa_states = pd.read_csv(StringIO(usa_states))
df_usa_states.rename(columns={'state':'state_name'},inplace=True)
df_usa_states.info()
if "code" not in accident_2016.columns:
    accident_2016 = accident_2016.merge(df_usa_states,how="outer",on="state_name")

df_usa = df_usa_states.merge(df_usa_population,on='state_name',how='inner')
df_usa = df_usa.set_index('code')
meaningless_cols = ['state_number','consecutive_number','county','city']
meaningless_cols.extend(['day_of_crash','minute_of_notification','month_of_crash','year_of_crash','day_of_week',
                        'hour_of_crash','minute_of_crash','national_highway_system',
                        'hour_of_notification','hour_of_arrival_at_scene','minute_of_arrival_at_scene',
                         'hour_of_ems_arrival_at_hospital','minute_of_ems_arrival_at_hospital'])

#accident_2016.groupby('state_name').sum().drop(meaningless_cols,axis=1).head(10)[['number_of_fatalities','number_of_drunk_drivers']]
filter_cols = ['number_of_fatalities','number_of_drunk_drivers']
accident_2016.groupby('code').sum().head(10)[filter_cols]
day_hour_fatality = pd.pivot_table(accident_2016[accident_2016['hour_of_crash']<24],
               values='number_of_fatalities',
               index=['day_of_week'],
               columns='hour_of_crash',
               fill_value = 0,
               aggfunc=np.sum)

fig, ax = plt.subplots(figsize=(16,4))    
cmap = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=False)
sns.heatmap(day_hour_fatality,linewidths=.5,linecolor='white',
            fmt='g',annot=True,annot_kws={"size": 10},
           cmap=cmap)
plt.title('Fatalities on 2016')
plt.xlabel('Hour of crash')
plt.ylabel('Day of week')
plt.show()
filter_cols = ['number_of_fatalities','number_of_drunk_drivers']
df = accident_2016.groupby('code').sum()[filter_cols]

plot_on_usa_map(df,loc=df.index.to_series(),data_column='number_of_fatalities',bar_title="Fatalities 2016",layout_title='Deaths',text="")
filter_cols = ['number_of_fatalities','number_of_drunk_drivers']
df = accident_2016.groupby('code').sum()[filter_cols]

plot_on_usa_map(df,loc=df.index.to_series(),data_column='number_of_drunk_drivers',bar_title="Drunks",layout_title='Drunk Killers',text="")
temp = pd.DataFrame()
temp['fatalities_100k']=df['number_of_fatalities']/df_usa['population']*100000
temp['drunks_100k']=df['number_of_drunk_drivers']/df_usa['population']*100000
plot_on_usa_map(temp,loc=df.index.to_series(),data_column='fatalities_100k',bar_title="Fatalities",layout_title='Fatalities by 100.000 people',text="")
print('Top 10 States on Fatalities by 100k people:')
temp.sort_values(by='fatalities_100k',ascending=False)['fatalities_100k'].head(10)
print('Botton 10 States on Fatalities by 100k people:')
temp.sort_values(by='fatalities_100k',ascending=True)['fatalities_100k'].head(10)
t = temp.sort_values(by='fatalities_100k',ascending=False)['fatalities_100k'].head(10).mean()
b= temp.sort_values(by='fatalities_100k',ascending=True)['fatalities_100k'].head(10).mean()
print(f'Top 10 States average are {t/b:.1f} times greater the botton States on Fatalities by 100.000 people')
plot_on_usa_map(temp,loc=df.index.to_series(),data_column='drunks_100k',bar_title="Drunks",layout_title='Drunk Drivers by 100.000 people',text="")
print('Top 10 States Drunk Drivers by 100k people:')
temp.sort_values(by='drunks_100k',ascending=False)['drunks_100k'].head(10)
print('Botton 10 States Drunk Drivers by 100k people:')
temp.sort_values(by='drunks_100k',ascending=True)['drunks_100k'].head(10)
t = temp.sort_values(by='drunks_100k',ascending=False)['drunks_100k'].head(10).mean()
b= temp.sort_values(by='drunks_100k',ascending=True)['drunks_100k'].head(10).mean()
print(f'Top 10 States average are {t/b:.1f} times greater the botton States on Drunk Drivers by 100.000 people')
sns.jointplot(x='drunks_100k',y='fatalities_100k',data=temp,kind='reg')
# To do: Plot a heatmap on all US (not grouping by State)
# Analyze 