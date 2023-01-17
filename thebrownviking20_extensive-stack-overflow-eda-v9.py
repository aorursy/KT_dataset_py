# Importing libraries
import numpy as np # For linear algebra
import pandas as pd # Python's extensive data analyis toolkit
# Matolotlib for visualizations
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
%matplotlib inline
# Seaborn for statistical insights
import seaborn as sns
sns.set_style('whitegrid')
# PLotly libraries for interactive visualizations
from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
# Squarify for treemaps
import squarify
# Random for well, random stuff
import random
# operator for sorting dictionaries
import operator
# For ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# More libraries may be added in further updates
# Importing data
data = pd.read_csv('../input/survey_results_public.csv')

data.describe()
data.head()
total = data.isnull().sum().sort_values(ascending = False)
percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False).astype('int')
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data['Column_Name'] = missing_data.index
trace = go.Table(
    header=dict(values=['Column_Name','Total','Percent'],
                fill = dict(color='#C2D4FF'),
                align = ['left'] * 5),
    cells=dict(values=[missing_data.Column_Name,missing_data.Total,missing_data.Percent],
               fill = dict(color='#F5F8FF'),
               align = ['left'] * 5))

d = [trace] 
iplot(d, filename = 'pandas_table')
# Preparing data and handling missing values
data['Student'] = data['Student'].fillna('Not known')
data["Country"] = data["Country"].fillna('Not known')
data["DevType"] = data["DevType"].fillna('Not mentioned')

data["Employment"][data["Employment"] == "Employed full-time"] = "Full-time"
data["Employment"][data["Employment"] == "Independent contractor, freelancer, or self-employed"] = "Self-employed"
data["Employment"][data["Employment"] == "Not employed, but looking for work"] = "Job-hunter"

data["Employment"][data["Employment"] == "Employed part-time"] = "Part-time"
data["Employment"][data["Employment"] == "Not employed, and not looking for work"] = "Not employed"
data["Employment"] = data["Employment"].fillna("Not known")

data["FormalEducation"][data["FormalEducation"] == "Bachelor’s degree (BA, BS, B.Eng., etc.)"] = "Bachelor's degree"
data["FormalEducation"][data["FormalEducation"] == "Master’s degree (MA, MS, M.Eng., MBA, etc.)"] = "Master's degree"
data["FormalEducation"][data["FormalEducation"] == "Some college/university study without earning a degree"] = "In college"
data["FormalEducation"][data["FormalEducation"] == "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)"] = "Secondary school"
data["FormalEducation"][data["FormalEducation"] == "Other doctoral degree (Ph.D, Ed.D., etc.)"] = "Doctoral"
data["FormalEducation"][data["FormalEducation"] == "Primary/elementary school"] = "ElementaryEd"
data["FormalEducation"][data["FormalEducation"] == "Professional degree (JD, MD, etc.)"] = "Professional"
data["FormalEducation"][data["FormalEducation"] == "I never completed any formal education"] = "No formal ed"
data["FormalEducation"][data["FormalEducation"] == "Associate degree"] = "Associate"
data["FormalEducation"] = data["FormalEducation"].fillna("Not known")

data["TimeAfterBootcamp"][data["TimeAfterBootcamp"] == "I already had a full-time job as a developer when I began the program"] = "Already had a job"


color_brewer = ['#57B8FF','#B66D0D','#009FB7','#FBB13C','#FE6847','#4FB5A5','#8C9376','#F29F60','#8E1C4A','#85809B','#515B5D','#9EC2BE','#808080','#9BB58E','#5C0029','#151515','#A63D40','#E9B872','#56AA53','#CE6786','#449339','#2176FF','#348427','#671A31','#106B26','008DD5','#034213','#BC2F59','#939C44','#ACFCD9','#1D3950','#9C5414','#5DD9C1','#7B6D49','#8120FF','#F224F2','#C16D45','#8A4F3D','#616B82','#443431','#340F09']
gs = gridspec.GridSpec(2, 2)

fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(gs[0, 0]) # row 0, col 0
sns.factorplot("Hobby",data=data,kind="count",ax=ax1,palette=sns.color_palette("Spectral", 10))

ax2 = fig.add_subplot(gs[0, 1]) # row 0, col 1
sns.factorplot("Hobby",data=data,kind="count",ax=ax2,hue="OpenSource",palette=sns.color_palette("cubehelix_r", 10))

ax3 = fig.add_subplot(gs[1, :]) # row 1, span all columns
sns.factorplot("Hobby",data=data,kind="count",ax=ax3,hue="Student",palette=sns.color_palette("Paired_r", 10))

plt.close(2)
plt.close(3)
plt.close(4)
plt.savefig('plot1.png')
gs = gridspec.GridSpec(2, 2)

fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(gs[0, 0]) # row 0, col 0
sns.factorplot("OpenSource",data=data,kind="count",ax=ax1,palette=sns.color_palette("Spectral_r", 10))

ax2 = fig.add_subplot(gs[0, 1]) # row 0, col 1
sns.factorplot("OpenSource",data=data,kind="count",ax=ax2,hue="Hobby",palette=sns.color_palette("cubehelix_r", 10))

ax3 = fig.add_subplot(gs[1, :]) # row 1, span all columns
sns.factorplot("OpenSource",data=data,kind="count",ax=ax3,hue="Student",palette=sns.color_palette("Paired_r", 10))

plt.close(2)
plt.close(3)
plt.close(4)
plt.savefig('plot2.png')
def treemap(v):
    x = 0.
    y = 0.
    width = 50.
    height = 50.
    type_list = v.index
    values = v.values

    normed = squarify.normalize_sizes(values, width, height)
    rects = squarify.squarify(normed, x, y, width, height)

    # Choose colors from http://colorbrewer2.org/ under "Export"
    shapes = []
    annotations = []
    counter = 0

    for r in rects:
        shapes.append( 
            dict(
                type = 'rect', 
                x0 = r['x'], 
                y0 = r['y'], 
                x1 = r['x']+r['dx'], 
                y1 = r['y']+r['dy'],
                line = dict( width = 1 ),
                fillcolor = color_brewer[counter]
            ) 
        )
        annotations.append(
            dict(
                x = r['x']+(r['dx']/2),
                y = r['y']+(r['dy']/2),
                text = "{}".format(type_list[counter]),
                showarrow = False
            )
        )
        counter = counter + 1
        if counter >= len(color_brewer):
            counter = 0

    # For hover text
    trace0 = go.Scatter(
        x = [ r['x']+(r['dx']/2) for r in rects ], 
        y = [ r['y']+(r['dy']/2) for r in rects ],
        text = [ str(v) for v in values ], 
        mode = 'text',
    )

    layout = dict(
        height=600, 
        width=800,
        xaxis=dict(showgrid=False,zeroline=False),
        yaxis=dict(showgrid=False,zeroline=False),
        shapes=shapes,
        annotations=annotations,
        hovermode='closest',
        font=dict(color="#FFFFFF")
    )

    # With hovertext
    figure = dict(data=[trace0], layout=layout)
    iplot(figure, filename='squarify-treemap')
country = data["Country"].dropna()
for i in country.unique():
    if country[country == i].count() < 600:
        country[country == i] = 'Others'
x = 0.
y = 0.
width = 50.
height = 50.
type_list = country.value_counts().index
values = country.value_counts().values

normed = squarify.normalize_sizes(values, width, height)
rects = squarify.squarify(normed, x, y, width, height)

# Choose colors from http://colorbrewer2.org/ under "Export"
color_brewer = color_brewer
shapes = []
annotations = []
counter = 0

for r in rects:
    shapes.append( 
        dict(
            type = 'rect', 
            x0 = r['x'], 
            y0 = r['y'], 
            x1 = r['x']+r['dx'], 
            y1 = r['y']+r['dy'],
            line = dict( width = 1 ),
            fillcolor = color_brewer[counter]
        ) 
    )
    annotations.append(
        dict(
            x = r['x']+(r['dx']/2),
            y = r['y']+(r['dy']/2),
            text = "{}".format(type_list[counter]),
            showarrow = False
        )
    )
    counter = counter + 1
    if counter >= len(color_brewer):
        counter = 0

# For hover text
trace0 = go.Scatter(
    x = [ r['x']+(r['dx']/2) for r in rects ], 
    y = [ r['y']+(r['dy']/2) for r in rects ],
    text = [ str(v) for v in values ], 
    mode = 'text',
)

layout = dict(
    height=600, 
    width=850,
    xaxis=dict(showgrid=False,zeroline=False),
    yaxis=dict(showgrid=False,zeroline=False),
    shapes=shapes,
    annotations=annotations,
    hovermode='closest',
    font=dict(color="#FFFFFF"),
    margin = go.Margin(
            l=0,
            r=0,
            pad=0
        )
)

# With hovertext
figure = dict(data=[trace0], layout=layout)
iplot(figure, filename='squarify-treemap')
treemap(data["Employment"].value_counts())
treemap(data["FormalEducation"].value_counts())
fig = {
  "data": [
    {
      "values": data["CompanySize"].value_counts().values,
      "labels": data["CompanySize"].value_counts().index,
      "domain": {"x": [0, .95]},
      "name": "Company size distribution",
      "hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie",
      "marker": {"colors": [i for i in reversed(color_brewer)]},
      "textfont": {"color": "#FFFFFF"}
    }],
  "layout": {
        "title":"Company size distribution",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Company Size",
                "x": 0.47,
                "y": 0.5
            }
        ]
    }
}
iplot(fig, filename='donut')
random.shuffle(color_brewer)

fig = {
  "data": [
    {
      "values": data["UndergradMajor"].value_counts().values,
      "labels": data["UndergradMajor"].value_counts().index,
      "domain": {"x": [0, .25]},
      "name": "Company size distribution",
      "hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie",
      "marker": {"colors": [i for i in reversed(color_brewer)]},
      "textfont": {"color": "#FFFFFF"}
    }],
  "layout": {
        "title":"Undergraduate education distribution",
        "paper_bgcolor": "#D3D3D3",
        "plot_bgcolor": "#D3D3D3",
        "annotations": [
            {
                "font": {
                    "size": 15
                },
                "showarrow": False,
                "text": "Course",
                "x": 0.08,
                "y": 0.5
            }
        ]
    }
}
iplot(fig, filename='donut')
def voteSimplifier(v,o,mar,t):
    d = {}
    for type in v:
        type = str(type).split(';')
        for i in type: 
            if i in d:
                d[i] = d[i] + 1
            else:
                d[i] = 0
    d = sorted(d.items(), key=operator.itemgetter(1))
    if o == 'v':
        d = list(reversed(d))
    d = dict(d)
    if o == 'v':
        Y = list(d.values())
        X = list(d.keys())
    else: 
        Y = list(d.keys())
        X = list(d.values())
    trace = [go.Bar(
                y=Y,
                x=X,
                orientation = o,
                marker=dict(color=color_brewer, line=dict(color='rgb(8,48,107)',width=1.5,)),
                opacity = 0.6
    )]
    layout = go.Layout(
        margin = go.Margin(
            l = mar
        ),
        title=t,
    )

    fig = go.Figure(data=trace, layout=layout)
    iplot(fig, filename='horizontal-bar')
random.shuffle(color_brewer)
voteSimplifier(data["DevType"].dropna(),'h',300,"Types of developers")
random.shuffle(color_brewer)
voteSimplifier(data["LanguageWorkedWith"].dropna(),'v',50,"Most used languages")
random.shuffle(color_brewer)
voteSimplifier(data["LanguageDesireNextYear"].dropna(),'v',50,"Languages developers want to use")
random.shuffle(color_brewer)
voteSimplifier(data["DatabaseWorkedWith"].dropna(),'v',50,"Most used databases")
random.shuffle(color_brewer)
voteSimplifier(data["DatabaseDesireNextYear"].dropna(),'v',50,"Databases developers want to use")
random.shuffle(color_brewer)
voteSimplifier(data["EducationTypes"].dropna(),'h',520,"Methods used by developers use to learn new technologies")
random.shuffle(color_brewer)
voteSimplifier(data["SelfTaughtTypes"].dropna(),'h',600,"How developers self-study")
random.shuffle(color_brewer)
voteSimplifier(data["PlatformWorkedWith"].dropna(),'v',50,"Most used platforms")
random.shuffle(color_brewer)
voteSimplifier(data["PlatformDesireNextYear"].dropna(),'v',50,"Platforms developers want to use")
random.shuffle(color_brewer)
voteSimplifier(data["FrameworkWorkedWith"].dropna(),'v',50,"Most used frameworks")
random.shuffle(color_brewer)
voteSimplifier(data["FrameworkDesireNextYear"].dropna(),'v',50,"Frameworks developers want to use")
random.shuffle(color_brewer)
voteSimplifier(data["IDE"].dropna(),'v',50,"Most popular IDEs")
fig = {
  "data": [
    {
      "values": data["OperatingSystem"].dropna().value_counts().values,
      "labels": data["OperatingSystem"].dropna().value_counts().index,
      "domain": {"x": [0, .95]},
      "name": "Operating System",
      "hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie",
      "marker": {"colors": [i for i in reversed(color_brewer)]},
      "textfont": {"color": "#FFFFFF"}
    }],
  "layout": {
        "title":"What operating systems are the developers using?",
        "paper_bgcolor": "#D3D3D3",
        "plot_bgcolor": "#D3D3D3",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Operating Systems",
                "x": 0.47,
                "y": 0.5
            }
        ]
    }
}
iplot(fig, filename='donut')
random.shuffle(color_brewer)
fig = {
  "data": [
    {
      "values": data["NumberMonitors"].dropna().value_counts().values,
      "labels": data["NumberMonitors"].dropna().value_counts().index,
      "domain": {"x": [0, .95]},
      "name": "Monitors",
      "hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie",
      "marker": {"colors": [i for i in reversed(color_brewer)]},
      "textfont": {"color": "#FFFFFF"}
    }],
  "layout": {
        "title":"Number of monitors used by developers",
        "paper_bgcolor": "#D3D3D3",
        "plot_bgcolor": "#D3D3D3",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Number of monitors",
                "x": 0.47,
                "y": 0.5
            }
        ]
    }
}
iplot(fig, filename='donut')
random.shuffle(color_brewer)
voteSimplifier(data["Methodology"].dropna(),'h',500,"Most popular programming methodologies")
random.shuffle(color_brewer)
voteSimplifier(data["VersionControl"].dropna(),'h',275,"Most popular version control systems")
random.shuffle(color_brewer)
fig = {
  "data": [
    {
      "values": data["CheckInCode"].dropna().value_counts().values,
      "labels": data["CheckInCode"].dropna().value_counts().index,
      "domain": {"x": [0, .95]},
      "name": "Checking code",
      "hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie",
      "marker": {"colors": [i for i in reversed(color_brewer)]},
      "textfont": {"color": "#FFFFFF"}
    }],
  "layout": {
        "title":"How often do developers check or commit code?",
        "paper_bgcolor": "#D3D3D3",
        "plot_bgcolor": "#D3D3D3",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Check/Commit",
                "x": 0.47,
                "y": 0.5
            }
        ]
    }
}
iplot(fig, filename='donut')
random.shuffle(color_brewer)
voteSimplifier(data["HackathonReasons"].dropna(),'h',600,"Why developers participate in hackathons?")
random.shuffle(color_brewer)
voteSimplifier(data["TimeAfterBootcamp"].dropna(),'h',200,"How much time after the bootcamp did the developers get jobs")
random.shuffle(color_brewer)
voteSimplifier(data["CommunicationTools"].dropna(),'h',400,"Most popular communication tools among developers")
treemap(data["YearsCoding"].dropna().value_counts())
treemap(data["YearsCodingProf"].dropna().value_counts())
random.shuffle(color_brewer)
fig = {
  "data": [
    {
      "values": data["JobSatisfaction"].dropna().value_counts().values,
      "labels": data["JobSatisfaction"].dropna().value_counts().index,
      "domain": {"x": [0, .95]},
      "name": "Job Satisfaction",
      "hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie",
      "marker": {"colors": [i for i in reversed(color_brewer)]},
      "textfont": {"color": "#FFFFFF"}
    }],
  "layout": {
        "title":"Are developers satisfied with their jobs?",
        "paper_bgcolor": "#D3D3D3",
        "plot_bgcolor": "#D3D3D3",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Job Satisfaction",
                "x": 0.47,
                "y": 0.5
            }
        ]
    }
}
iplot(fig, filename='donut')
random.shuffle(color_brewer)
fig = {
  "data": [
    {
      "values": data["CareerSatisfaction"].dropna().value_counts().values,
      "labels": data["CareerSatisfaction"].dropna().value_counts().index,
      "domain": {"x": [0, .95]},
      "name": "Career Satisfaction",
      "hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie",
      "marker": {"colors": [i for i in reversed(color_brewer)]},
      "textfont": {"color": "#FFFFFF"}
    }],
  "layout": {
        "title":"Are developers satisfied with their career?",
        "paper_bgcolor": "#D3D3D3",
        "plot_bgcolor": "#D3D3D3",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Career Satisfaction",
                "x": 0.47,
                "y": 0.5
            }
        ]
    }
}
iplot(fig, filename='donut')
random.shuffle(color_brewer)
fig = {
  "data": [
    {
      "values": data["HopeFiveYears"].dropna().value_counts().values,
      "labels": data["HopeFiveYears"].dropna().value_counts().index,
      "domain": {"x": [0, .95]},
      "name": "Hopes for future",
      "hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie",
      "marker": {"colors": [i for i in reversed(color_brewer)]},
      "textfont": {"color": "#FFFFFF"}
    }],
  "layout": {
        "title":"What developers want to be doing in next 5 years?",
        "paper_bgcolor": "#D3D3D3",
        "plot_bgcolor": "#D3D3D3",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Aspirations",
                "x": 0.47,
                "y": 0.5
            }
        ],
        "legend": dict(orientation="h")
    }
}
iplot(fig, filename='donut')
random.shuffle(color_brewer)
fig = {
  "data": [
    {
      "values": data["JobSearchStatus"].dropna().value_counts().values,
      "labels": data["JobSearchStatus"].dropna().value_counts().index,
      "domain": {"x": [0, .95]},
      "name": "Hopes for future",
      "hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie",
      "marker": {"colors": [i for i in reversed(color_brewer)]},
      "textfont": {"color": "#FFFFFF"}
    }],
  "layout": {
        "title":"Description of job-seeking status of developers",
        "paper_bgcolor": "#D3D3D3",
        "plot_bgcolor": "#D3D3D3",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Job seeking",
                "x": 0.47,
                "y": 0.5
            }
        ],
        "legend": dict(orientation="h")
    }
}
iplot(fig, filename='donut')
random.shuffle(color_brewer)
fig = {
  "data": [
    {
      "values": data["LastNewJob"].dropna().value_counts().values,
      "labels": data["LastNewJob"].dropna().value_counts().index,
      "domain": {"x": [0, .95]},
      "name": "Last job",
      "hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie",
      "marker": {"colors": [i for i in reversed(color_brewer)]},
      "textfont": {"color": "#FFFFFF"}
    }],
  "layout": {
        "title":"When was the last time a developer took a job with a new employer?",
        "paper_bgcolor": "#D3D3D3",
        "plot_bgcolor": "#D3D3D3",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Time since last job",
                "x": 0.47,
                "y": 0.5
            }
        ]
    }
}
iplot(fig, filename='donut')
random.shuffle(color_brewer)
fig = {
  "data": [
    {
      "values": data["UpdateCV"].dropna().value_counts().values,
      "labels": data["UpdateCV"].dropna().value_counts().index,
      "domain": {"x": [0, .95]},
      "name": "UpdateCV",
      "hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie",
      "marker": {"colors": [i for i in reversed(color_brewer)]},
      "textfont": {"color": "#FFFFFF"}
    }],
  "layout": {
        "title":"Last time CV was updated...",
        "paper_bgcolor": "#D3D3D3",
        "plot_bgcolor": "#D3D3D3",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Last CV update",
                "x": 0.47,
                "y": 0.5
            }
        ],
        "legend":dict(orientation="h")
    }
}
iplot(fig, filename='donut')
random.shuffle(color_brewer)
fig = {
  "data": [
    {
      "values": data["SalaryType"].dropna().value_counts().values,
      "labels": data["SalaryType"].dropna().value_counts().index,
      "domain": {"x": [0, .95]},
      "name": "Salary Type",
      "hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie",
      "marker": {"colors": [i for i in reversed(color_brewer)]},
      "textfont": {"color": "#FFFFFF"}
    }],
  "layout": {
        "title":"At what frequency are the developers paid?",
        "paper_bgcolor": "#D3D3D3",
        "plot_bgcolor": "#D3D3D3",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Salary Type",
                "x": 0.47,
                "y": 0.5
            }
        ]
    }
}
iplot(fig, filename='donut')
fig = ff.create_distplot([data["ConvertedSalary"].dropna()],['Converted Salary'],bin_size=10000)
iplot(fig, filename='Basic Distplot')
factor_list = ["The industry that I'd be working in","The financial performance or funding status of the company or organization","The specific department or team I'd be working on","The languages, frameworks, and other technologies I'd be working with","The compensation and benefits offered","The office environment or company culture","The opportunity to work from home/remotely","Opportunities for professional development","The diversity of the company or organization","How widely used or impactful the product or service I'd be working on is"]
mean_list = [data["AssessJob{}".format(i)].dropna().mean() for i in range(1,11)]
assess_job = pd.DataFrame()
assess_job["Factors"] = factor_list
assess_job["Mean_Score"] = mean_list
assess_job["Rank"] = assess_job["Mean_Score"].rank()
df = assess_job.sort_values("Rank")

trace1 = go.Table(
    header=dict(values=df.columns,
                fill = dict(color='#C2D4FF'),
                align = ['left'] * 5),
    cells=dict(values=[df.Factors, df.Mean_Score, df.Rank],
               fill = dict(color='#F5F8FF'),
               align = ['left'] * 5))

d = [trace1]

iplot(d, filename = 'pandas_table')
factor_list = ["Salary and/or bonuses","Stock options or shares","Health insurance","Parental leave","Fitness or wellness benefit","Retirement or pension savings","Company-provided meals or snacks","Computer/office equipment allowance","Childcare benefit","Transportation benefit","Conference or education budget"]
mean_list = [data["AssessBenefits{}".format(i)].dropna().mean() for i in range(1,12)]
assess_benefits = pd.DataFrame()
assess_benefits["Factors"] = factor_list
assess_benefits["Mean_Score"] = mean_list
assess_benefits["Rank"] = assess_benefits["Mean_Score"].rank()
df = assess_benefits.sort_values("Rank")

trace1 = go.Table(
    header=dict(values=df.columns,
                fill = dict(color='#AC68CC'),
                align = ['left'] * 5),
    cells=dict(values=[df.Factors, df.Mean_Score, df.Rank],
               fill = dict(color='#D6B4E7'),
               align = ['left'] * 5))

d = [trace1]

iplot(d, filename = 'pandas_table')
random.shuffle(color_brewer)

d1 = {}
for type in data["LanguageWorkedWith"][data.OpenSource == "Yes"].dropna():
    type = str(type).split(';')
    for i in type: 
        if i in d1:
            d1[i] = d1[i] + 1
        else:
            d1[i] = 0
d1 = sorted(d1.items(), key=operator.itemgetter(1))
d1 = dict(d1)

d2 = {}
for type in data["LanguageWorkedWith"][data.OpenSource == "No"].dropna():
    type = str(type).split(';')
    for i in type: 
        if i in d2:
            d2[i] = d2[i] + 1
        else:
            d2[i] = 0
d2 = sorted(d2.items(), key=operator.itemgetter(1))
d2 = dict(d2)

trace1 = go.Bar(
    x = list(d1.values()),
    y = list(d1.keys()),
    orientation='h',
    marker=dict(color=color_brewer, line=dict(color='rgb(8,48,107)',width=1.5,)),
    opacity = 0.6
)

trace2 = go.Bar(
    x = list(d2.values()),
    y = list(d2.keys()),
    orientation='h',
    marker=dict(color=color_brewer, line=dict(color='rgb(8,48,107)',width=1.5,)),
    opacity = 0.6
)

fig = tools.make_subplots(rows=1, cols=2, shared_yaxes=True)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)

fig['layout'].update(height=1000, width=900, title='Open source contributors vs non-open-source contributors(Most used languages)', showlegend=False, xaxis=dict(domain=[0, 0.41],autorange="reversed"), xaxis2=dict(domain=[0.59, 1]), yaxis=dict(side='right'), yaxis2=dict(side='right'))
iplot(fig, filename='simple-subplot')
trace1 = go.Bar(
    y=data["FormalEducation"][data["JobSatisfaction"] == "Extremely dissatisfied"].value_counts().index,
    x=data["FormalEducation"][data["JobSatisfaction"] == "Extremely dissatisfied"].value_counts().values,
    name='Extremely dissatisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(246, 78, 139, 0.6)',
        line = dict(
            color = 'rgba(246, 78, 139, 1.0)',
            width = 3)
    )
)
trace2 = go.Bar(
    y=data["FormalEducation"][data["JobSatisfaction"] == "Moderately dissatisfied"].value_counts().index,
    x=data["FormalEducation"][data["JobSatisfaction"] == "Moderately dissatisfied"].value_counts().values,
    name='Moderately dissatisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(58, 71, 80, 0.6)',
        line = dict(
            color = 'rgba(58, 71, 80, 1.0)',
            width = 3)
    )
)
trace3 = go.Bar(
    y=data["FormalEducation"][data["JobSatisfaction"] == "Slightly dissatisfied"].value_counts().index,
    x=data["FormalEducation"][data["JobSatisfaction"] == "Slightly dissatisfied"].value_counts().values,
    name='Slightly dissatisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(255, 225, 79, 0.6)',
        line = dict(
            color = 'rgba(255, 225, 79, 1.0)',
            width = 3)
    )
)
trace4 = go.Bar(
    y=data["FormalEducation"][data["JobSatisfaction"] == "Neither satisfied nor dissatisfied"].value_counts().index,
    x=data["FormalEducation"][data["JobSatisfaction"] == "Neither satisfied nor dissatisfied"].value_counts().values,
    name='Neither satisfied nor dissatisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(180, 49, 49, 0.6)',
        line = dict(
            color = 'rgba(180, 49, 49, 1.0)',
            width = 3)
    )
)
trace5 = go.Bar(
    y=data["FormalEducation"][data["JobSatisfaction"] == "Slightly satisfied"].value_counts().index,
    x=data["FormalEducation"][data["JobSatisfaction"] == "Slightly satisfied"].value_counts().values,
    name='Slightly satisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(49, 102, 191, 0.6)',
        line = dict(
            color = 'rgba(49, 102, 191, 1.0)',
            width = 3)
    )
)
trace6 = go.Bar(
    y=data["FormalEducation"][data["JobSatisfaction"] == "Moderately satisfied"].value_counts().index,
    x=data["FormalEducation"][data["JobSatisfaction"] == "Moderately satisfied"].value_counts().values,
    name='Moderately satisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(245, 157, 22, 0.6)',
        line = dict(
            color = 'rgba(245, 157, 22, 1.0)',
            width = 3)
    )
)
trace7 = go.Bar(
    y=data["FormalEducation"][data["JobSatisfaction"] == "Extremely satisfied"].value_counts().index,
    x=data["FormalEducation"][data["JobSatisfaction"] == "Extremely satisfied"].value_counts().values,
    name='Extremely satisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(158, 251, 71, 0.6)',
        line = dict(
            color = 'rgba(158, 251, 71, 1.0)',
            width = 3)
    )
)

d = [trace1, trace2,trace3,trace4,trace5,trace6,trace7]
layout = go.Layout(
    barmode='stack',
    margin=go.Margin(
        l=120
    ),
    title="Effect of formal education on job satisfaction"
)

fig = go.Figure(data=d, layout=layout)
iplot(fig, filename='marker-h-bar')
trace1 = go.Bar(
    y=data["CompanySize"][data["JobSatisfaction"] == "Extremely dissatisfied"].value_counts().index,
    x=data["CompanySize"][data["JobSatisfaction"] == "Extremely dissatisfied"].value_counts().values,
    name='Extremely dissatisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(246, 78, 139, 0.6)',
        line = dict(
            color = 'rgba(246, 78, 139, 1.0)',
            width = 3)
    )
)
trace2 = go.Bar(
    y=data["CompanySize"][data["JobSatisfaction"] == "Moderately dissatisfied"].value_counts().index,
    x=data["CompanySize"][data["JobSatisfaction"] == "Moderately dissatisfied"].value_counts().values,
    name='Moderately dissatisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(58, 71, 80, 0.6)',
        line = dict(
            color = 'rgba(58, 71, 80, 1.0)',
            width = 3)
    )
)
trace3 = go.Bar(
    y=data["CompanySize"][data["JobSatisfaction"] == "Slightly dissatisfied"].value_counts().index,
    x=data["CompanySize"][data["JobSatisfaction"] == "Slightly dissatisfied"].value_counts().values,
    name='Slightly dissatisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(255, 225, 79, 0.6)',
        line = dict(
            color = 'rgba(255, 225, 79, 1.0)',
            width = 3)
    )
)
trace4 = go.Bar(
    y=data["CompanySize"][data["JobSatisfaction"] == "Neither satisfied nor dissatisfied"].value_counts().index,
    x=data["CompanySize"][data["JobSatisfaction"] == "Neither satisfied nor dissatisfied"].value_counts().values,
    name='Neither satisfied nor dissatisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(180, 49, 49, 0.6)',
        line = dict(
            color = 'rgba(180, 49, 49, 1.0)',
            width = 3)
    )
)
trace5 = go.Bar(
    y=data["CompanySize"][data["JobSatisfaction"] == "Slightly satisfied"].value_counts().index,
    x=data["CompanySize"][data["JobSatisfaction"] == "Slightly satisfied"].value_counts().values,
    name='Slightly satisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(49, 102, 191, 0.6)',
        line = dict(
            color = 'rgba(49, 102, 191, 1.0)',
            width = 3)
    )
)
trace6 = go.Bar(
    y=data["CompanySize"][data["JobSatisfaction"] == "Moderately satisfied"].value_counts().index,
    x=data["CompanySize"][data["JobSatisfaction"] == "Moderately satisfied"].value_counts().values,
    name='Moderately satisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(245, 157, 22, 0.6)',
        line = dict(
            color = 'rgba(245, 157, 22, 1.0)',
            width = 3)
    )
)
trace7 = go.Bar(
    y=data["CompanySize"][data["JobSatisfaction"] == "Extremely satisfied"].value_counts().index,
    x=data["CompanySize"][data["JobSatisfaction"] == "Extremely satisfied"].value_counts().values,
    name='Extremely satisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(158, 251, 71, 0.6)',
        line = dict(
            color = 'rgba(158, 251, 71, 1.0)',
            width = 3)
    )
)

d = [trace1, trace2,trace3,trace4,trace5,trace6,trace7]
layout = go.Layout(
    barmode='stack',
    margin=go.Margin(
        l=200
    ),
    title="Effect of company size on job satisfaction"
)

fig = go.Figure(data=d, layout=layout)
iplot(fig, filename='marker-h-bar')
trace1 = go.Bar(
    y=data["HopeFiveYears"][data["JobSatisfaction"] == "Extremely dissatisfied"].value_counts().index,
    x=data["HopeFiveYears"][data["JobSatisfaction"] == "Extremely dissatisfied"].value_counts().values,
    name='Extremely dissatisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(246, 78, 139, 0.6)',
        line = dict(
            color = 'rgba(246, 78, 139, 1.0)',
            width = 3)
    )
)
trace2 = go.Bar(
    y=data["HopeFiveYears"][data["JobSatisfaction"] == "Moderately dissatisfied"].value_counts().index,
    x=data["HopeFiveYears"][data["JobSatisfaction"] == "Moderately dissatisfied"].value_counts().values,
    name='Moderately dissatisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(58, 71, 80, 0.6)',
        line = dict(
            color = 'rgba(58, 71, 80, 1.0)',
            width = 3)
    )
)
trace3 = go.Bar(
    y=data["HopeFiveYears"][data["JobSatisfaction"] == "Slightly dissatisfied"].value_counts().index,
    x=data["HopeFiveYears"][data["JobSatisfaction"] == "Slightly dissatisfied"].value_counts().values,
    name='Slightly dissatisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(255, 225, 79, 0.6)',
        line = dict(
            color = 'rgba(255, 225, 79, 1.0)',
            width = 3)
    )
)
trace4 = go.Bar(
    y=data["HopeFiveYears"][data["JobSatisfaction"] == "Neither satisfied nor dissatisfied"].value_counts().index,
    x=data["HopeFiveYears"][data["JobSatisfaction"] == "Neither satisfied nor dissatisfied"].value_counts().values,
    name='Neither satisfied nor dissatisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(180, 49, 49, 0.6)',
        line = dict(
            color = 'rgba(180, 49, 49, 1.0)',
            width = 3)
    )
)
trace5 = go.Bar(
    y=data["HopeFiveYears"][data["JobSatisfaction"] == "Slightly satisfied"].value_counts().index,
    x=data["HopeFiveYears"][data["JobSatisfaction"] == "Slightly satisfied"].value_counts().values,
    name='Slightly satisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(49, 102, 191, 0.6)',
        line = dict(
            color = 'rgba(49, 102, 191, 1.0)',
            width = 3)
    )
)
trace6 = go.Bar(
    y=data["HopeFiveYears"][data["JobSatisfaction"] == "Moderately satisfied"].value_counts().index,
    x=data["HopeFiveYears"][data["JobSatisfaction"] == "Moderately satisfied"].value_counts().values,
    name='Moderately satisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(245, 157, 22, 0.6)',
        line = dict(
            color = 'rgba(245, 157, 22, 1.0)',
            width = 3)
    )
)
trace7 = go.Bar(
    y=data["HopeFiveYears"][data["JobSatisfaction"] == "Extremely satisfied"].value_counts().index,
    x=data["HopeFiveYears"][data["JobSatisfaction"] == "Extremely satisfied"].value_counts().values,
    name='Extremely satisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(158, 251, 71, 0.6)',
        line = dict(
            color = 'rgba(158, 251, 71, 1.0)',
            width = 3)
    )
)

d = [trace1, trace2,trace3,trace4,trace5,trace6,trace7]
layout = go.Layout(
    barmode='stack',
    margin=go.Margin(
        l=500
    ),
    title="Effect of job satisfaction on future goals",
    legend=dict(orientation="h")
)

fig = go.Figure(data=d, layout=layout)
iplot(fig, filename='marker-h-bar')
trace1 = go.Bar(
    y=data["YearsCoding"][data["JobSatisfaction"] == "Extremely dissatisfied"].value_counts().index,
    x=data["YearsCoding"][data["JobSatisfaction"] == "Extremely dissatisfied"].value_counts().values,
    name='Extremely dissatisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(246, 78, 139, 0.6)',
        line = dict(
            color = 'rgba(246, 78, 139, 1.0)',
            width = 3)
    )
)
trace2 = go.Bar(
    y=data["YearsCoding"][data["JobSatisfaction"] == "Moderately dissatisfied"].value_counts().index,
    x=data["YearsCoding"][data["JobSatisfaction"] == "Moderately dissatisfied"].value_counts().values,
    name='Moderately dissatisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(58, 71, 80, 0.6)',
        line = dict(
            color = 'rgba(58, 71, 80, 1.0)',
            width = 3)
    )
)
trace3 = go.Bar(
    y=data["YearsCoding"][data["JobSatisfaction"] == "Slightly dissatisfied"].value_counts().index,
    x=data["YearsCoding"][data["JobSatisfaction"] == "Slightly dissatisfied"].value_counts().values,
    name='Slightly dissatisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(255, 225, 79, 0.6)',
        line = dict(
            color = 'rgba(255, 225, 79, 1.0)',
            width = 3)
    )
)
trace4 = go.Bar(
    y=data["YearsCoding"][data["JobSatisfaction"] == "Neither satisfied nor dissatisfied"].value_counts().index,
    x=data["YearsCoding"][data["JobSatisfaction"] == "Neither satisfied nor dissatisfied"].value_counts().values,
    name='Neither satisfied nor dissatisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(180, 49, 49, 0.6)',
        line = dict(
            color = 'rgba(180, 49, 49, 1.0)',
            width = 3)
    )
)
trace5 = go.Bar(
    y=data["YearsCoding"][data["JobSatisfaction"] == "Slightly satisfied"].value_counts().index,
    x=data["YearsCoding"][data["JobSatisfaction"] == "Slightly satisfied"].value_counts().values,
    name='Slightly satisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(49, 102, 191, 0.6)',
        line = dict(
            color = 'rgba(49, 102, 191, 1.0)',
            width = 3)
    )
)
trace6 = go.Bar(
    y=data["YearsCoding"][data["JobSatisfaction"] == "Moderately satisfied"].value_counts().index,
    x=data["YearsCoding"][data["JobSatisfaction"] == "Moderately satisfied"].value_counts().values,
    name='Moderately satisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(245, 157, 22, 0.6)',
        line = dict(
            color = 'rgba(245, 157, 22, 1.0)',
            width = 3)
    )
)
trace7 = go.Bar(
    y=data["YearsCoding"][data["JobSatisfaction"] == "Extremely satisfied"].value_counts().index,
    x=data["YearsCoding"][data["JobSatisfaction"] == "Extremely satisfied"].value_counts().values,
    name='Extremely satisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(158, 251, 71, 0.6)',
        line = dict(
            color = 'rgba(158, 251, 71, 1.0)',
            width = 3)
    )
)

d = [trace1, trace2,trace3,trace4,trace5,trace6,trace7]
layout = go.Layout(
    barmode='stack',
    margin=go.Margin(
        l=150
    ),
    title="Effect of coding experience on job satisfaction"
)

fig = go.Figure(data=d, layout=layout)
iplot(fig, filename='marker-h-bar')
trace1 = go.Bar(
    y=data["Age"][data["JobSatisfaction"] == "Extremely dissatisfied"].value_counts().index,
    x=data["Age"][data["JobSatisfaction"] == "Extremely dissatisfied"].value_counts().values,
    name='Extremely dissatisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(246, 78, 139, 0.6)',
        line = dict(
            color = 'rgba(246, 78, 139, 1.0)',
            width = 3)
    )
)
trace2 = go.Bar(
    y=data["Age"][data["JobSatisfaction"] == "Moderately dissatisfied"].value_counts().index,
    x=data["Age"][data["JobSatisfaction"] == "Moderately dissatisfied"].value_counts().values,
    name='Moderately dissatisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(58, 71, 80, 0.6)',
        line = dict(
            color = 'rgba(58, 71, 80, 1.0)',
            width = 3)
    )
)
trace3 = go.Bar(
    y=data["Age"][data["JobSatisfaction"] == "Slightly dissatisfied"].value_counts().index,
    x=data["Age"][data["JobSatisfaction"] == "Slightly dissatisfied"].value_counts().values,
    name='Slightly dissatisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(255, 225, 79, 0.6)',
        line = dict(
            color = 'rgba(255, 225, 79, 1.0)',
            width = 3)
    )
)
trace4 = go.Bar(
    y=data["Age"][data["JobSatisfaction"] == "Neither satisfied nor dissatisfied"].value_counts().index,
    x=data["Age"][data["JobSatisfaction"] == "Neither satisfied nor dissatisfied"].value_counts().values,
    name='Neither satisfied nor dissatisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(180, 49, 49, 0.6)',
        line = dict(
            color = 'rgba(180, 49, 49, 1.0)',
            width = 3)
    )
)
trace5 = go.Bar(
    y=data["Age"][data["JobSatisfaction"] == "Slightly satisfied"].value_counts().index,
    x=data["Age"][data["JobSatisfaction"] == "Slightly satisfied"].value_counts().values,
    name='Slightly satisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(49, 102, 191, 0.6)',
        line = dict(
            color = 'rgba(49, 102, 191, 1.0)',
            width = 3)
    )
)
trace6 = go.Bar(
    y=data["Age"][data["JobSatisfaction"] == "Moderately satisfied"].value_counts().index,
    x=data["Age"][data["JobSatisfaction"] == "Moderately satisfied"].value_counts().values,
    name='Moderately satisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(245, 157, 22, 0.6)',
        line = dict(
            color = 'rgba(245, 157, 22, 1.0)',
            width = 3)
    )
)
trace7 = go.Bar(
    y=data["Age"][data["JobSatisfaction"] == "Extremely satisfied"].value_counts().index,
    x=data["Age"][data["JobSatisfaction"] == "Extremely satisfied"].value_counts().values,
    name='Extremely satisfied',
    orientation = 'h',
    marker = dict(
        color = 'rgba(158, 251, 71, 0.6)',
        line = dict(
            color = 'rgba(158, 251, 71, 1.0)',
            width = 3)
    )
)

d = [trace1,trace2,trace3,trace4,trace5,trace6,trace7]
layout = go.Layout(
    barmode='stack',
    margin=go.Margin(
        l=150
    ),
    title="Effect of age on job satisfaction"
)

fig = go.Figure(data=d, layout=layout)
iplot(fig, filename='marker-h-bar')
trace1 = {"x": [data["ConvertedSalary"][data["Country"]==i].mean() for i in data["Country"].value_counts().index],
          "y": data["Country"].value_counts().index,
          "marker": {"color": "pink", "size": 12},
          "mode": "markers",
          "name": "Mean Salary",
          "type": "scatter"
}

d = [trace1]
layout = {"title": "Mean salary in different countries",
          "xaxis": {"title": "Converted Salary", },
          "yaxis": {"title": "Country"},
          "height":3500,
          "margin":dict(l=300)
         }

fig = go.Figure(data=d, layout=layout)
iplot(fig, filename='basic_dot-plot')
random.shuffle(color_brewer)
fig = {
  "data": [
    {
      "values": data["EthicsChoice"].dropna().value_counts().values,
      "labels": data["EthicsChoice"].dropna().value_counts().index,
      "domain": {"x": [0, .95]},
      "name": "Unethical code",
      "hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie",
      "marker": {"colors": [i for i in reversed(color_brewer)]},
      "textfont": {"color": "#FFFFFF"}
    }],
  "layout": {
        "title":"Do developers write code for unethical purposes/projects?",
        "paper_bgcolor": "#D3D3D3",
        "plot_bgcolor": "#D3D3D3",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Unethical code",
                "x": 0.47,
                "y": 0.5
            }
        ]
    }
}
iplot(fig, filename='donut')
d = [go.Bar(
            x=['Ethical','Unethical'],
            y=[data["ConvertedSalary"][data["EthicsChoice"]=="No"].mean(),data["ConvertedSalary"][data["EthicsChoice"]=="Yes"].mean()],
            text=[int(data["ConvertedSalary"][data["EthicsChoice"]=="No"].mean()),int(data["ConvertedSalary"][data["EthicsChoice"]=="Yes"].mean())],
            textposition='auto',
            marker=dict(
                color=['rgb(158,202,225)','rgb(58,200,225)'],
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5,
                )
            ),
            opacity=0.6
    )]
layout = go.Layout(
    title="Are good ethics rewarding(mean salaries of ethical and unethical developers)?",
    margin=go.Margin(
        l=100,
        r=100
    )
)
fig = go.Figure(data=d, layout=layout)
iplot(fig, filename='bar')
random.shuffle(color_brewer)
fig = {
  "data": [
    {
      "values": data["EthicsReport"].dropna().value_counts().values,
      "labels": data["EthicsReport"].dropna().value_counts().index,
      "domain": {"x": [0, .95]},
      "name": "Report/Call out",
      "hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie",
      "marker": {"colors": [i for i in reversed(color_brewer)]},
      "textfont": {"color": "#FFFFFF"}
    }],
  "layout": {
        "title":"Do developers report these unethical practices?",
        "paper_bgcolor": "#D3D3D3",
        "plot_bgcolor": "#D3D3D3",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Report/Call out",
                "x": 0.47,
                "y": 0.5
            }
        ]
    }
}
iplot(fig, filename='donut')
random.shuffle(color_brewer)
fig = {
  "data": [
    {
      "values": data["EthicsResponsible"].dropna().value_counts().values,
      "labels": data["EthicsResponsible"].dropna().value_counts().index,
      "domain": {"x": [0, .95]},
      "name": "Who is responsible",
      "hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie",
      "marker": {"colors": [i for i in reversed(color_brewer)]},
      "textfont": {"color": "#FFFFFF"}
    }],
  "layout": {
        "title":"Who is responsible for success of unethical code?",
        "paper_bgcolor": "#D3D3D3",
        "plot_bgcolor": "#D3D3D3",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Who is responsible?",
                "x": 0.47,
                "y": 0.5
            }
        ]
    }
}
iplot(fig, filename='donut')
random.shuffle(color_brewer)
fig = {
  "data": [
    {
      "values": data["AIDangerous"].dropna().value_counts().values,
      "labels": data["AIDangerous"].dropna().value_counts().index,
      "domain": {"x": [0, .95]},
      "name": "AIDangerous",
      "hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie",
      "marker": {"colors": [i for i in reversed(color_brewer)]},
      "textfont": {"color": "#FFFFFF"}
    }],
  "layout": {
        "title":"Dangerous aspects of AI",
        "paper_bgcolor": "#D3D3D3",
        "plot_bgcolor": "#D3D3D3",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Dangerous AI",
                "x": 0.47,
                "y": 0.5
            }
        ],
        "legend": dict(orientation="h")
    }
}
iplot(fig, filename='donut')
random.shuffle(color_brewer)
fig = {
  "data": [
    {
      "values": data["AIInteresting"].dropna().value_counts().values,
      "labels": data["AIInteresting"].dropna().value_counts().index,
      "domain": {"x": [0, .95]},
      "name": "AIInteresting",
      "hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie",
      "marker": {"colors": [i for i in reversed(color_brewer)]},
      "textfont": {"color": "#FFFFFF"}
    }],
  "layout": {
        "title":"Interesting aspects of AI",
        "paper_bgcolor": "#D3D3D3",
        "plot_bgcolor": "#D3D3D3",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Interesting AI",
                "x": 0.47,
                "y": 0.5
            }
        ],
        "legend": dict(orientation="h")
    }
}
iplot(fig, filename='donut')
random.shuffle(color_brewer)
fig = {
  "data": [
    {
      "values": data["AIResponsible"].dropna().value_counts().values,
      "labels": data["AIResponsible"].dropna().value_counts().index,
      "domain": {"x": [0, .95]},
      "name": "Who will take the responsibilities of AI's effect?",
      "hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie",
      "marker": {"colors": [i for i in reversed(color_brewer)]},
      "textfont": {"color": "#FFFFFF"}
    }],
  "layout": {
        "title":"Who will take the responsibilities of AI's effect?",
        "paper_bgcolor": "#D3D3D3",
        "plot_bgcolor": "#D3D3D3",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Responsible for AI",
                "x": 0.47,
                "y": 0.5
            }
        ],
        "legend": dict(orientation="h")
    }
}
iplot(fig, filename='donut')
random.shuffle(color_brewer)
fig = {
  "data": [
    {
      "values": data["AIFuture"].dropna().value_counts().values,
      "labels": data["AIFuture"].dropna().value_counts().index,
      "domain": {"x": [0, .95]},
      "name": "AIFuture",
      "hoverinfo":"label+percent+name",
      "hole": .7,
      "type": "pie",
      "marker": {"colors": [i for i in reversed(color_brewer)]},
      "textfont": {"color": "#FFFFFF"}
    }],
  "layout": {
        "title":"The future of AI",
        "paper_bgcolor": "#D3D3D3",
        "plot_bgcolor": "#D3D3D3",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Future of AI",
                "x": 0.47,
                "y": 0.5
            }
        ],
        "legend": dict(orientation="h")
    }
}
iplot(fig, filename='donut')