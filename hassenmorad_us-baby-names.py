import csv

import json

import re

import numpy as np

import pandas as pd

import altair as alt



from collections import Counter, OrderedDict

from IPython.display import HTML

from  altair.vega import v3
# This whole section 

vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v3.SCHEMA_VERSION

vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'

vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION

vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'

noext = "?noext"



paths = {

    'vega': vega_url + noext,

    'vega-lib': vega_lib_url + noext,

    'vega-lite': vega_lite_url + noext,

    'vega-embed': vega_embed_url + noext

}



workaround = """

requirejs.config({{

    baseUrl: 'https://cdn.jsdelivr.net/npm/',

    paths: {}

}});

"""



#------------------------------------------------ Defs for future rendering

def add_autoincrement(render_func):

    # Keep track of unique <div/> IDs

    cache = {}

    def wrapped(chart, id="vega-chart", autoincrement=True):

        if autoincrement:

            if id in cache:

                counter = 1 + cache[id]

                cache[id] = counter

            else:

                cache[id] = 0

            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])

        else:

            if id not in cache:

                cache[id] = 0

            actual_id = id

        return render_func(chart, id=actual_id)

    # Cache will stay outside and 

    return wrapped

            

@add_autoincrement

def render(chart, id="vega-chart"):

    chart_str = """

    <div id="{id}"></div><script>

    require(["vega-embed"], function(vg_embed) {{

        const spec = {chart};     

        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);

        console.log("anything?");

    }});

    console.log("really...anything?");

    </script>

    """

    return HTML(

        chart_str.format(

            id=id,

            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)

        )

    )



HTML("".join((

    "<script>",

    workaround.format(json.dumps(paths)),

    "</script>",

    "This code block sets up embedded rendering in HTML output and<br/>",

    "provides the function `render(chart, id='vega-chart')` for use below."

)))
state_pops = pd.read_csv('../input/historical-state-populations-19002017/state_pops.csv')

state_names = pd.read_csv('../input/us-state-baby-names/all_states.csv')

nat_names = pd.read_csv('../input/us-national-baby-names-18802017/all_national.csv')
name_variability = {}

for year in range(1880,2018):

    df = nat_names[nat_names['Year'] == year]

    variability = sum(df.Count)/len(df)

    name_variability[year] = round(variability)
name_count = pd.DataFrame({'Year':list(name_variability.keys()), 

                           'Count':list(name_variability.values())})

yrs = name_count['Year'].astype('str')

name_count['Year'] = pd.to_datetime(yrs)
base = alt.Chart(name_count).encode(

    x=alt.X('year(Year)', axis=alt.Axis(grid=True), title='Year'),

    y=alt.Y('Count', axis=alt.Axis(grid=False))

    ).properties(width=600, title='Average Count Per Name')



line = base.mark_line()



points = base.mark_point().encode(

    tooltip=['year(Year)', 'Count'])



chart = line + points

render(chart)
#Dictionary containing all names that appeared in national top 10 in any year during a specified decade

#Split into: M & F

def top_10_potential(decade):

    all_names = {}

    for sex in ['M','F']:

        names = []

        

        #Adding conditional to avoid error w/ 2010 decade data (since data only covers 2010-2017)

        if decade == 2010:

            span = 8

        else:

            span = 10

            

        #Extracting top 10 names from each year within a decade

        for i in range(span):

            df = nat_names[nat_names['Year'] == decade+i].reset_index(drop=True)

            df = df[df['Sex'] == sex].reset_index(drop=True)

            top_10 = list(df.loc[:9, 'Name'])

            new = [name for name in top_10 if name not in names]

            names += new

        #Stores list of potential names in relevant key (M or F) 

        all_names[sex] = names

    return all_names
#Returns list of decade's top 10 names (based on cumulative count)

#(First parameter should be variable storing output of top_10_potential function- specifying M or F key)

def top_10_confirm(decade):

    potential = top_10_potential(decade)

    counts = {}

    

    #Conditional to avoid error w/ 2010 decade data (since data only covers 2010-2017)

    if decade == 2010:

        span = 8

    else:

        span = 10

        

    for sex in ['M','F']:

        names = {}

        for year in range(decade, decade+span):

            df = nat_names[nat_names['Year'] == year].reset_index(drop=True)

            for name in potential[sex]:

                #Try/except averts error if name has no count in a particular year

                try:

                    count = df['Count'][(df['Sex'] == sex) & (df['Name'] == name)].reset_index(drop=True).iloc[0]

                except Exception: 

                    count = 0

                    

                #Creates key for each name and assigns annual count in names dict 

                #(or increments count value for existing name key, doing this for every year in decade to arrive at total count)

                if name in list(names.keys()):

                    names[name] += count

                else:

                    names[name] = count

        #Sorts names dict by values (ascending) and stores keys of top 10 names (corresponding to a particular sex) in counts dict

        counts[sex] = sorted(names, key=names.get)[-10:]

    return counts
#Dictionary containing all names that were one of the 10 most popular names nationally in any decade (1910-2020)

#(not including 1880-1900 since state name data begins from 1910, which this dict was created to examine)

#(split into M & F)

trended = {}

decades = list(np.arange(1910, 2020, 10))

for sex in ['M','F']:

    top_10 = []

    for decade in decades:

        names = top_10_confirm(decade)

        new = [name for name in names[sex] if name not in top_10]

        top_10 += new

    trended[sex] = top_10
#Populating dictionary w/ names that appeared in top 10 from any year, categorized by sex & decade

#Name keys are assigned empty values, which I will manually replace w/ syllable count below

all_names_decades = {}

decades = list(np.arange(1880, 2020, 10))

for sex in ['M','F']:

    names = {}

    for decade in decades:

        keys = top_10_potential(decade)[sex] 

        names[decade] = {key: None for key in keys} 

    all_names_decades[sex] = names
#Dictionary containing unique male & female names from all_names_decades dictionary 

#I will manually assign syllable count to names in dict below

unique_names = {}

decades = list(np.arange(1880,2020,10))

for sex in ['M','F']:

    original = []

    for decade in decades:

        names = top_10_potential(decade)[sex]

        new = [name for name in names if name not in original]

        original += new

    unique_names[sex] = {key: None for key in original}
#Copy/pasted contents of unique_names (above) and manually assigned syllable counts to new dict (original_names)

#For names whose syllable counts I was unsure about, I assigned a syllable count 

#consistent w/ popular pronunciation (e.g. Barbara(2), Amelia(3), Brittany(2), Deborah(2))

original_names = {

'F': {'Abigail': 3, 'Addison': 3,

  'Alexis': 2,  'Alice': 2, 'Alyssa': 3, 'Amanda': 3, 'Amelia': 3, 'Amy': 2, 'Angela': 3, 'Anna': 2,

  'Annie': 2, 'Bertha': 2, 'Bessie': 2, 'Clara': 2, 'Edna': 2, 'Ethel': 2, 'Florence': 2, 'Gladys': 2,

  'Ida': 2, 'Lillian': 3, 'Minnie': 2, 'Ashley': 2, 'Ava': 2, 'Barbara': 2,  'Betty': 2, 'Brenda': 2,  

  'Brittany': 2, 'Carol': 2, 'Carolyn': 3, 'Charlotte': 2,  'Chloe': 2, 'Crystal': 2,  'Cynthia': 3, 

  'Deborah': 2,  'Debra': 2, 'Donna': 2,  'Doris': 2, 'Dorothy': 3, 'Elizabeth': 4, 'Emily': 3, 'Emma': 2,

  'Evelyn': 3, 'Frances': 2, 'Hannah': 2, 'Harper': 2, 'Heather': 2, 'Helen': 2, 'Isabella': 4, 

  'Jennifer': 3, 'Jessica': 3, 'Joan': 1, 'Joyce': 1, 'Judith': 2, 'Judy': 2, 'Julie': 2, 'Karen': 2,  

  'Kathleen': 2, 'Kayla': 2,  'Kelly': 2, 'Kimberly': 3, 'Laura': 2, 'Lauren': 2,  'Linda': 2, 'Lisa': 2,

  'Lori': 2, 'Madison': 3, 'Margaret': 2, 'Marie': 2, 'Mary': 2, 'Megan': 2, 'Melissa': 3, 'Mia': 2, 

  'Michelle': 2, 'Mildred': 2, 'Nancy': 2, 'Nicole': 2, 'Olivia': 4, 'Pamela': 3, 'Patricia': 3, 

  'Rachel': 2, 'Rebecca': 3, 'Ruth': 1, 'Samantha': 3, 'Sandra': 2, 'Sarah': 2, 'Sharon': 2,'Shirley': 2,

  'Sophia': 3, 'Stephanie': 3, 'Susan': 2, 'Tammy': 2, 'Taylor': 2, 'Tracy': 2, 'Virginia': 3},

  

 'M': {'Aiden': 2, 'Alexander': 4, 'Andrew': 2, 'Anthony': 3, 'Austin': 2, 'Benjamin': 3, 'Brandon': 2, 

  'Brian': 2,'Charles': 2, 'Christopher': 3, 'Daniel': 2, 'David': 2, 'Donald': 2, 'Edward': 2,

  'Elijah': 3, 'Ethan': 2, 'Frank': 1, 'Gary': 2, 'George': 1, 'Harry': 2, 'Henry': 2, 'Jacob': 2, 

  'James': 1, 'Jason': 2, 'Jayden': 2, 'Jeffrey': 2, 'John': 1, 'Joseph': 2, 'Joshua': 3, 'Justin': 2, 

  'Larry': 2, 'Liam': 2, 'Logan': 2, 'Mark': 1, 'Mason': 2, 'Matthew': 2, 'Michael': 2, 'Nicholas': 3,  

  'Noah': 2, 'Oliver': 3, 'Richard': 2, 'Robert': 2, 'Ronald': 2, 'Scott': 1, 'Steven': 2, 'Thomas': 2, 

  'Tyler': 2, 'Walter': 2, 'William': 2}}
#Transfers syllable count from original_names dict to corresponding names 

#(w/ empty values) in all_names_decades dict

for sex in ['M','F']:

    decades = np.arange(1880, 2020, 10)

    for decade in decades:

        dec = all_names_decades[sex][decade]

        for name in dec:

            all_names_decades[sex][decade][name] = original_names[sex][name]

            

#Proceeding With New Dict (after syllable count assignment)

decade_syllable_counts = all_names_decades.copy()            
#Stores avg syllable count for all names appearing in top 10 for a particular year in each decade

syllables_avg = {}

decades = np.arange(1880,2020,10)

for sex in ['F', 'M']:

    avgs = {}

    for decade in decades:

        data = decade_syllable_counts[sex][decade]

        avg = np.mean(list(data.values()))

        avgs[decade] = avg

    syllables_avg[sex] = avgs
# Dataframe for Altair plot

male = pd.DataFrame.from_dict(syllables_avg['M'], orient='index')

male['Sex'] = np.full(len(male),'M')



female = pd.DataFrame.from_dict(syllables_avg['F'], orient='index')

female['Sex'] = np.full(len(female),'F')



combined_syllables = pd.concat([male, female])

combined_syllables = combined_syllables.reset_index().rename({'index':'Year', 0:'Syllables'}, axis=1)

combined_syllables['Syllables'] = combined_syllables['Syllables'].round(2)



combined_syllables['Year'] = pd.DatetimeIndex(combined_syllables['Year'].astype('str'))
base = alt.Chart(combined_syllables).encode(

    x=alt.X('year(Year)', axis=alt.Axis(grid=True), title=None),

    y=alt.Y('Syllables:Q', scale=alt.Scale(zero=False), axis=alt.Axis(grid=False)),

    color=alt.Color(

        'Sex', scale=alt.Scale(

        domain=['F','M'], 

        range=['#de9ed6','#1f77b4']))

    ).properties(width=600, height=400, title='Average Syllable Count (for popular names)')



lines = base.mark_line()

points = base.mark_point().encode(tooltip=['year(Year)','Syllables'])



chart = lines + points

render(chart)
#Dictionary storing peak year for each name in trended (for all states)

#(split into M & F)

#Need to find faster way to process this (took 2 min 10 sec)

male_fem = {}

state_abrev = list(state_names.State.unique())

for sex in ['M','F']:

    names = {}

    for name in trended[sex]:

        states = {}

        copy = state_names.copy()

        copy = copy[(copy['Sex'] == sex) & (copy['Name'] == name)]

        for state in state_abrev:

            df = copy[copy['State'] == state]

            peak = df['Normalized_Count'].max()

            peak_yr = df['Year'][df['Normalized_Count'] == peak]

            states[state] = peak_yr.iloc[0]

        names[name] = states

    male_fem[sex] = names
abigail = male_fem['F']['Abigail']

points = pd.DataFrame({'State':list(abigail.keys()), 'Peak':list(abigail.values())})

dates = points.Peak.astype('str')

points['Peak'] = pd.to_datetime(dates)



click = alt.selection_multi(encodings=['color'])



pts = alt.Chart(points).mark_circle(size=80).encode(

    x=alt.X('State:O', title=''),

    y=alt.Y('year(Peak)', title='Peak Year'),

    color=alt.condition(click, 'State', alt.value('lightgray'), legend=None),

    tooltip=['State', 'year(Peak)']

).properties(selection=click, width=600, height=500, title='Year of Each State\'s Peak Popularity for "Abigail"')



legend = alt.Chart(points).mark_circle(size=80).encode(

    y=alt.Y('State:O', title=''),

    color=alt.condition(click, 'State', alt.value('lightgray'), legend=None)

).properties(height=600, selection=click)



chart = pts | legend

render(chart)
#Dictionary split into M & F#Diction 

state_counts = {}

for sex in ['M','F']:

    count = {}

    for name in male_fem[sex]:

        data = male_fem[sex][name]

        values = list(data.items())

        peaks = list(data.values())

        first_peak = min(peaks)

        

        #Identifying trend-setting states whose peaks were within 10 years of median peak

        median = np.median(peaks)

        for peak in peaks:

            if median - first_peak <= 10:

                trend_setters = [x[0] for x in values if x[1] == first_peak]

                break

        #Removing outliers from peaks list & identifying next earliest non-outlier peak

            else:

                peaks = list(filter(lambda x: x!= first_peak, peaks))

                first_peak = min(peaks)

                continue

        #Populating counts dict w/ counts for states that have earliest peak appearance

        for state in trend_setters:

            if state in count:

                count[state] += 1

            else:

                count[state] = 1

        #Storing value of 0 for states w/ no earliest peak appearances

        states = [x[0] for x in values]

        for state in states:

            if state not in count:

                count[state] = 0

    state_counts[sex] = count
#Combining counts from M & F sub-dictionaries into combined counts dictionary

male = state_counts['M']

female = state_counts['F']

state_counts['Combined'] = {x: male.get(x, 0) + female.get(x, 0) 

                            for x in set(male) & set(female)}



counts = pd.DataFrame({'Boys':state_counts['M'], 'Girls':state_counts['F'], 'Combined':state_counts['Combined']})

counts = counts.reset_index().rename({'index':'State'}, axis=1)
#Creating 'Count' series containing 51 'male' & 51 'female' labels (to add to df below)

male_state_counts = counts.loc[:,['State','Boys']]

female_state_counts = counts.loc[:,['State','Girls']]



male_state_counts['Sex'] = np.full(51, 'Boys')

male_state_counts.rename({'Boys':'Count'}, axis=1, inplace=True)

female_state_counts['Sex'] = np.full(51, 'Girls')

female_state_counts.rename({'Girls':'Count'}, axis=1, inplace=True)



both_counts = pd.concat([male_state_counts, female_state_counts]).reset_index(drop=True)



#Creating 'combined_counts_repeat' series (containing combined male+female counts, repeated) 

#to assign to new 'Total' column below

combined_counts = both_counts.groupby('State').Count.sum()

combined_counts_repeat = pd.concat([combined_counts, combined_counts])

both_counts['Total'] = combined_counts_repeat.reset_index(drop=True)
bar = alt.Chart(both_counts).mark_bar().encode(

    x=alt.X('State', title=None),

    y=alt.Y('Total', title=None),

    color=(alt.condition(        

        alt.datum.Total > 11,

        alt.value('orange'),

        alt.value('steelblue'))),

    tooltip=['State','Total']

    ).properties(width=700, title='Total # of Earliest Peaks')



render(bar)
bar = alt.Chart(both_counts).mark_bar(opacity=0.9).encode(

    x=alt.X('State', title=None),

    y=alt.Y('Count', title=None),

    tooltip=['State','Count','Total'],

    color=alt.Color('Sex', 

                    scale = alt.Scale(domain=['Boys', 'Girls'],

                  range=['#386cb0', '#e377c2']))

    ).properties(width=650, title='Total # of Earliest Peaks')



render(bar)
#Storing avg years before/after national median peak for popular names (for each state)

avg_diff = {}

for sex in ['M', 'F']:

    peaks_diff = {}

    temp={}

    data = male_fem[sex]

    for name in data:

        peaks = list(data[name].values())

        median = np.median(peaks)

        state_peak_pairs = list(data[name].items())

        for pair in state_peak_pairs:

            state = pair[0]

            peak = pair[1]

            

#IMPORTANT: Filtering out outliers (States whose peaks were more than 5 years before national median peak)

#*Tweaking the number below (5) will affect what peak year data is included 

# (since it discards peak years occurring more than 5 yrs before national median)



            if abs(peak-median) > 5:

                continue

            #appending median-peak differences to peaks_diff dictionary    

            elif state in peaks_diff:

                peaks_diff[state] += [peak-median]

            else:

                peaks_diff[state] = [peak-median]

    #Calculating average peak-median difference for each state 

    #(negative value indicates avg # of years it peaks before median peak)

    for state in peaks_diff:

        temp[state] = np.mean(peaks_diff[state])

    #Storing avg differences for each sex (in avg_diff dict)

    avg_diff[sex] = temp
av_dif = pd.DataFrame([avg_diff['M'], avg_diff['F']], index=['Boys','Girls'])

av_dif = av_dif.transpose()

av_dif['Combined'] = (av_dif.Boys + av_dif.Girls) /2

av_dif = av_dif.round(2)

av_dif = av_dif.reset_index().rename({'index':'State'}, axis=1)
bar = alt.Chart(av_dif).mark_bar().encode(

    x=alt.X('State', title=None),

    y=alt.Y('Combined', title='Years'),

    color=alt.Color('Combined', scale = alt.Scale(range=['indigo', 'teal']), legend=None),

    tooltip=['State','Boys','Girls', 'Combined']

    ).properties(width=700, title="Avg. Peak Relative to National Median")



render(bar)
import plotly

plotly.offline.init_notebook_mode()

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



data = dict(

    type='choropleth',

    locations=av_dif['State'],

    locationmode='USA-states',

    colorscale = 'YlGnBu',

    z=av_dif['Combined'])



lyt = dict(geo=dict(scope='usa'), title='Avg. Peak Relative To National Median')

map = go.Figure(data=[data], layout=lyt)

plotly.offline.iplot(map)