import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from urllib.request import urlopen
import json
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)  
import os 
print(os.listdir("../input"))
medicare = pd.read_csv('../input/Medicare Hospital Spending by Claim.csv')
medicare.head()
medicare.info()
medicare.describe()
medicare['Period'].value_counts()
def changing_period_names(previous_name):
    if previous_name == '1 through 30 days After Discharge from Index Hospital Admission':
        return 'After Discharge'
    elif previous_name == 'During Index Hospital Admission':
        return 'During Hospital Admission'
    elif previous_name =='1 to 3 days Prior to Index Hospital Admission':
        return 'Prior Hospital Admission'
    elif previous_name == 'Complete Episode':
        return 'Overall'
    
medicare['Period'] = medicare['Period'].map(changing_period_names)
medicare['Period'].value_counts()
medicare['Claim Type'].value_counts()
medicare['Facility ID'].nunique()
import re
#Removing the % symbol
medicare['Percent of Spndg Hospital'] = medicare['Percent of Spndg Hospital'].apply(lambda x: re.sub('%', '', x))
medicare['Percent of Spndg State'] = medicare['Percent of Spndg State'].apply(lambda x: re.sub('%', '', x))
medicare['Percent of Spndg National'] = medicare['Percent of Spndg National'].apply(lambda x: re.sub('%', '', x))
#Converting into float
medicare['Percent of Spndg Hospital'] = medicare['Percent of Spndg Hospital'].apply(lambda x: float(x))
medicare['Percent of Spndg State'] = medicare['Percent of Spndg State'].apply(lambda x: float(x))
medicare['Percent of Spndg National'] = medicare['Percent of Spndg National'].apply(lambda x: float(x))
south_alabama = medicare[(medicare['Facility Name'] == 'SOUTHEAST ALABAMA MEDICAL CENTER') &(medicare['Claim Type'] != 'Total')]
#Since we are looking for all claim types, the claim type will be total
total = medicare[(medicare['Claim Type'] == 'Total')]
#We are dropping the start date and end date as we dont need it.
total.drop(columns = ['Start Date','End Date'], inplace = True)
#The total number of states that we have
total['State'].nunique()
All_States = total['State'].unique()
#These are all the states
print(All_States)
def data_avg_spend_hospital_statewise(df):
    avg_per_hospital = [] #Captures sum of the avg spending for EACH state e.g for AL = 12000 for AK = 12333
    
    total_avg_spend_hospital = df.agg({'Avg Spndg Per EP Hospital':sum})[0]#Captures sum of the avg spending for ALL state
   
    all_states = df['State'].unique()
    
    for i in all_states:
        state_value = df[(df['State'] == i)].agg({'Avg Spndg Per EP Hospital':sum})[0]
        average = (state_value/total_avg_spend_hospital) *100
        avg_per_hospital.append(average)
        average = 0
    
    new_dataframe = pd.DataFrame(list(zip(all_states, avg_per_hospital )),columns  = ['State','Spending'],index = [all_states])
    return new_dataframe
Statewise_Hosital = data_avg_spend_hospital_statewise(total)
top_ten = Statewise_Hosital.sort_values('Spending',ascending=False).head(n = 10)
# l= top_ten['State']
# sizes = top_ten['Spending']
# fig1, ax1 = plt.subplots()
# color_set = sns.color_palette("hls", 10)
# ax1.pie(sizes,labels=l, autopct='%1.1f%%',startangle = 90,  colors = color_set, radius = 1.8)
# plt.show()


fig = px.pie(top_ten,values = 'Spending', names = 'State',color_discrete_sequence=px.colors.sequential.RdBu,
            title='Top 10 States where Hospital wise spending is more')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
inpatient_cost = medicare[medicare['Claim Type'] == 'Inpatient']
outpatient_cost = medicare[medicare['Claim Type'] == 'Outpatient']
inpatient_cost.drop(columns = ['Facility ID','Start Date', 'End Date','Percent of Spndg Hospital','Percent of Spndg State','Percent of Spndg National'],inplace = True)
outpatient_cost.drop(columns = ['Facility ID','Start Date', 'End Date','Percent of Spndg Hospital','Percent of Spndg State','Percent of Spndg National'],inplace = True)
inpatient_state = pd.pivot_table(inpatient_cost,index=["State"])
outpatient_state = pd.pivot_table(outpatient_cost,index=["State"])
in_spending = inpatient_state.sort_values(by=['Avg Spndg Per EP Hospital','Avg Spndg Per EP State'], ascending = False)[:10]
out_spending = outpatient_state.sort_values(by=['Avg Spndg Per EP Hospital','Avg Spndg Per EP State'], ascending = False)[:10]
in_spending
in_spending_top_ten = pd.DataFrame(in_spending.to_records())
fig = px.bar(in_spending_top_ten,x = 'State', y=['Avg Spndg Per EP State','Avg Spndg Per EP Hospital'], title = 'Inpatient Spending w.r.t State and Hospital(top 10)',
        color_discrete_map={
                "Avg Spndg Per EP State": "grey",
                "Avg Spndg Per EP Hospital": "coral"})
fig.show()
out_spending 
out_spending_top_ten = pd.DataFrame(out_spending.to_records())
fig = go.Figure(data = [
     go.Bar(name = 'Avg Spndg Per EP State',x = out_spending_top_ten['State'], y = out_spending_top_ten['Avg Spndg Per EP State']),
     go.Bar(name = 'Avg Spndg Per EP Hospital',x = out_spending_top_ten['State'], y = out_spending_top_ten['Avg Spndg Per EP Hospital'])
        ]
        )
fig.update_layout(barmode='group', title = 'Outpatient spending w.r.t to States and Hospital (top 10 states)')
fig.show()
nursing = medicare[medicare['Claim Type'] == 'Skilled Nursing Facility']
nursing.drop(columns = ['Facility ID','Start Date','End Date','Facility Name'],inplace = True)
nursing.head()
nursing.Period.value_counts()
skilled_nursing_pivot = pd.pivot_table(nursing, index = 'State')
skilled_nursing_pivot_top_ten = skilled_nursing_pivot.sort_values(by = 'Avg Spndg Per EP State', ascending = False)[:10]
#Converting pivot table into dataframe
skilled_nursing_pivot_top_ten = pd.DataFrame(skilled_nursing_pivot_top_ten.to_records())
import plotly.graph_objects as go
fig = go.Figure(
    go.Bar(x = skilled_nursing_pivot_top_ten['Avg Spndg Per EP State'],
        y = skilled_nursing_pivot_top_ten['State'],
        orientation = 'h',
        marker=dict( color='teal'),
        name= 'Skilled nursing (Top Ten)'
    )
    )
fig.update_layout(title_text='Skilled Nursing (Top 10 states)')
fig.show()
medical_facility_center = pd.pivot_table(medicare.drop(columns = ['Facility ID','Percent of Spndg Hospital','Percent of Spndg National','Avg Spndg Per EP National','Percent of Spndg State']), index = ['Facility Name','State'])
medical_facility = pd.DataFrame(medical_facility_center.to_records())
medical_facility.sort_values(by =['Avg Spndg Per EP State'], ascending = False)[:10]
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    states = json.load(response)
import plotly.express as px
fig = px.choropleth(locations=medical_facility.State, title = 'Statewise spending done on medical centers',locationmode="USA-states", color = medical_facility['Avg Spndg Per EP State'],scope="usa")
fig.show()
pd.pivot_table(medicare, index = 'Period')