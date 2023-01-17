import math
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from datetime import datetime
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tools
py.init_notebook_mode(connected=True)
print("Provided datasets:\n")
loans = pd.read_csv("../input/kiva_loans.csv")
print("loans:\n", loans.shape)

mpi_region_locations = pd.read_csv("../input/kiva_mpi_region_locations.csv")
print("mpi_region_locations:\n", mpi_region_locations.shape)
loans.head()
print("Columns in the dataset:\n", sorted(loans.columns.tolist()))
loans_per_partner = loans.partner_id.value_counts()
loans_per_partner_no_outlier = loans_per_partner[loans_per_partner<100000]
fig = tools.make_subplots(rows=2, cols=2,
                          subplot_titles=["Loans per partner",  "Loans per partner (upper outlier removed)", 
                                          "", ""],
                          shared_xaxes=True,)

# Left side
trace1 = go.Histogram(x=loans_per_partner.tolist(), 
                      marker=dict(color="#FF851B"), showlegend=False)
trace2 = go.Box(x=loans_per_partner.tolist(), boxpoints='all', orientation='h', 
                marker=dict(color="#FF851B"), showlegend=False)
# Right side
trace3 = go.Histogram(x=loans_per_partner_no_outlier.tolist(), 
                      marker=dict(color="#3D9970"), showlegend=False)
trace4 = go.Box(x=loans_per_partner_no_outlier.tolist(), boxpoints='all', orientation='h', 
                marker=dict(color="#3D9970"), showlegend=False)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 1, 2)
fig.append_trace(trace4, 2, 2)

fig["layout"]["yaxis1"].update(dict(domain=[0.4, 1]))
fig["layout"]["yaxis2"].update(dict(domain=[0.4, 1]))
fig["layout"]["yaxis3"].update(dict(domain=[0, .4]), showticklabels=False)
fig["layout"]["yaxis4"].update(dict(domain=[0, .4]), showticklabels=False)

py.iplot(fig, filename="BasicStats")
num_partners = len(loans.partner_id.unique())
num_loans = len(loans)
print("{} partners applied for loans".format(num_partners))
print("{:.2f} mean loans per partner".format(np.mean(loans_per_partner)))
print("{:.2f} median loans per partner".format(np.median(loans_per_partner)))
def process_borrower_count(s):
    """
    Counts the number of genders listed in the `borrower_genders` column.
    If a borrower gender is unspecified, don't record the count.
    """
    borrower_list = [y.replace(" ", "") for y in s.split(",")]
    if len(borrower_list) == 1 and borrower_list[0]=="unknown":
        return None
    else:
        return len(borrower_list)

loans["borrower_count"] = loans["borrower_genders"].fillna("unknown").map(process_borrower_count)
unique_borrower_counts = loans.pivot_table(index="borrower_count", values="partner_id", aggfunc=lambda x: len(x.unique())).reset_index()

trace = go.Bar(x=unique_borrower_counts["borrower_count"].tolist(),
               y=unique_borrower_counts["partner_id"].tolist())
layout = dict(
    title="Size of Borrower Groups",
    xaxis=dict(title="Number of People in Partner Group"),
    yaxis=dict(title="Number of Groups")
)

fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig)
fig = tools.make_subplots(rows=2, cols=1,
                          subplot_titles=["Number of Loans to Groups of Each Size", "(with 550k single borrowers removed)"],
                          shared_xaxes=True,)

trace1 = go.Histogram(x=loans.borrower_count.dropna().tolist(), name="")
trace2 = go.Histogram(x=loans[loans["borrower_count"]>1].borrower_count.dropna().tolist(), name="")

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)

fig["layout"].update(showlegend=False,
                     width=1000, height=700,
                    xaxis1=dict(title="Number of People in Partner Groups"),
                    yaxis1=dict(title="Number of Loans"),
                    yaxis2=dict(title="Number of Loans"))

py.iplot(fig)
loan_terms_months = pd.DataFrame(loans.term_in_months.value_counts().sort_index()).reset_index()
loan_terms_months.columns = ["months", "count"]

loan_terms_years = loan_terms_months.copy()
loan_terms_years["months"] = np.ceil(loan_terms_years/12)
loan_terms_years.columns = ["years", "count"]
loan_terms_years = loan_terms_years.sort_values(by="years", ascending=True)
loan_terms_years = loan_terms_years.pivot_table(index="years", values="count", aggfunc=np.sum).reset_index()
loan_terms_years["years"] = loan_terms_years["years"].apply(lambda x: "{} - ".format(str(int(x-1))) + str(int(x)))
fig = tools.make_subplots(rows=2, cols=1,
                          subplot_titles=["Loan terms (months)",  "Loans terms (years)"],
                          shared_xaxes=False,)

traceMonths = go.Bar(x=loan_terms_months["months"].tolist(), 
                     y=loan_terms_months["count"].tolist(),
                    showlegend=False,
                    text=loan_terms_months["months"].tolist(),
                    name="")
traceYears = go.Bar(x=loan_terms_years["years"].tolist(),
                    y=loan_terms_years["count"].tolist(),
                    showlegend=False,
                    text=loan_terms_years["years"].tolist(),
                   name="")

fig.append_trace(traceMonths, 1, 1)
fig.append_trace(traceYears, 2, 1)

fig["layout"]["yaxis1"].update(dict(domain=[0.55, 1]))
fig["layout"]["yaxis2"].update(dict(domain=[0, 0.45]))
fig["layout"].update(dict(width=1000,
                         height=700),
                    xaxis1=dict(tickvals=list(range(0,168,12))),
                    xaxis2=dict(tickangle=-55))

py.iplot(fig, filename="LoanTerms")
def process_funded(row):
    """
    Classify loans into one of the categories described above.
    """
    requested = row["loan_amount"]
    funded = row["funded_amount"]
    diff = requested - funded
    if diff == 0:
        val = "Full funding"
    elif diff > 0 and funded != 0:
        val = "Partial funding"
    elif funded == 0:
        val = "Not funded"
    elif funded > requested:
        val = "Overfunded"
    else:
        val = "error"
    return val

loans["funded"] = loans.apply(lambda x: process_funded(x), axis=1)
funded_counts = loans["funded"].value_counts()
funded_total = np.sum(funded_counts)
funded_pcts = round(funded_counts/funded_total, 4)
names = funded_pcts.index.tolist()
vals = funded_pcts.values.tolist()
colors = ["green", "orange", "red", "gray"]
overall_data = list()
for i in range(len(names)):
    trace = go.Bar(
        y=["Overall"],
        x=[vals[i]],
        text=funded_counts.values.tolist()[i],
        name=names[i],
        orientation='h',
        marker=dict(color=colors[i])
    )
    overall_data.append(trace)

layout = go.Layout(
    barmode='stack',
    title="Funding Status of All Loans",
    width=1000, height=300,
)

fig = go.Figure(data=overall_data, layout=layout)
py.iplot(fig, filename='stacked-bar')
sectors = loans.sector.value_counts()

names = sectors.index.tolist()
vals = sectors.tolist()

sector_data = list()
for i in range(len(names)):
    trace = go.Bar(
        y=["Sectors"],
        x=[vals[i]],
#         text=vals[i],
        name=names[i],
        orientation='h',
    )
    sector_data.append(trace)

layout = go.Layout(
    barmode='stack',
    title="Number of Loans per Sector",
    xaxis=dict(title="Number of Loans"),
    height=465, width=1000
)
fig = go.Figure(data=sector_data, layout=layout)
py.iplot(fig)
sector_counts = loans.pivot_table(index="sector", columns="funded", values="id", aggfunc=lambda x: len(x)).fillna(0)

# Create new df which contains "pct of total" instead of "count":
sector_totals = np.sum(sector_counts, axis=1)
sector_pcts = sector_counts.copy()
for col in sector_counts.columns:
    sector_pcts[col] = round(sector_counts[col]/sector_totals, 4)

# Reorganize each dataframe:
sector_pcts = sector_pcts[["Full funding", "Partial funding", "Not funded", "Overfunded"]].sort_values(by="Full funding")
sector_counts = sector_counts[["Full funding", "Partial funding", "Not funded", "Overfunded"]].loc[sector_pcts.index]
sector_names = sector_pcts.index.tolist()
column_names = sector_pcts.columns.tolist()
colors = ["green", "orange", "red", "gray"]
sector_data = list()
for i in range(len(column_names)):
    trace = go.Bar(
        y=sector_names,
        x=sector_pcts[column_names[i]].tolist(),
        text=sector_counts[column_names[i]].tolist(),
        name=column_names[i],
        showlegend=False,
        orientation='h',
        marker=dict(color=colors[i])
    )
    sector_data.append(trace)

layout = go.Layout(
    barmode='stack',
    title="Percent funded (by Sector)"
)

# fig = go.Figure(data=sector_data, layout=layout)
# py.iplot(fig, filename='stacked-bar')
fig = tools.make_subplots(rows=2, cols=1,
                          subplot_titles=["Percent funded (by Sector)", ""],
                          shared_xaxes=True,)

for x in overall_data:
    fig.append_trace(x, 1, 1)
for x in sector_data:
    fig.append_trace(x, 2, 1)

fig["layout"].update(barmode='stack')
fig["layout"]["yaxis2"].update(dict(domain=[0, 0.7]))
fig["layout"]["yaxis1"].update(dict(domain=[0.8, 1]))
fig['layout'].update(height=800, width=1000)


py.iplot(fig, filename='customizing-subplot-axes')
# Genders
def process_genders(s):
    genders_list = list(set([y.replace(" ", "") for y in s.split(",")]))
    if len(genders_list) == 1:
        return genders_list[0]
    else:
        return "mixed"

loans["borrower_genders_processed"] = loans.borrower_genders.fillna("unknown").map(process_genders)
loan_partner_genders = loans.pivot_table(index="borrower_genders_processed", values="partner_id", aggfunc=lambda x: len(x.unique())).reset_index()
loan_genders = loans["borrower_genders_processed"].value_counts()

fig = tools.make_subplots(rows=2, cols=1,
                          subplot_titles=["Gender",""],
                          shared_xaxes=True,)

traceUnique = go.Bar(x=loan_partner_genders["borrower_genders_processed"],
                     y=loan_partner_genders["partner_id"],
                     marker=dict(color=['#FFCDD2', "#A2D5F2", "#59606D"]),
                     showlegend=False,
                    )
traceLoans = go.Bar(x=loan_genders.index,
                    y=loan_genders.values,
                    marker=dict(color=['#FFCDD2', "#A2D5F2", "#59606D"]),
                    showlegend=False,
                    )

fig.append_trace(traceUnique, 1, 1)
fig.append_trace(traceLoans, 2, 1)

fig["layout"].update(dict(width=1000,
                         height=700),
                    yaxis1=dict(title="Number of Partners",
                                domain=[0.55, 1]),
                    yaxis2=dict(title="Number of Loans",
                                domain=[0, 0.45]))

py.iplot(fig, filename="LoanGenders")
gender_counts = loans.pivot_table(index="borrower_genders_processed", columns="funded", values="id", aggfunc=lambda x: len(x)).fillna(0)

# Create new df which contains "pct of total" instead of "count":
gender_totals = np.sum(gender_counts, axis=1)
gender_pcts = gender_counts.copy()
for col in gender_counts.columns:
    gender_pcts[col] = round(gender_counts[col]/gender_totals, 4)

# Reorganize each dataframe:
gender_pcts = gender_pcts[["Full funding", "Partial funding", "Not funded", "Overfunded"]].sort_values(by="Full funding")
gender_counts = gender_counts[["Full funding", "Partial funding", "Not funded", "Overfunded"]].loc[gender_pcts.index]
gender_names = gender_pcts.index.tolist()
column_names = gender_pcts.columns.tolist()
colors = ["green", "orange", "red", "gray"]
gender_data = list()
for i in range(len(column_names)):
    trace = go.Bar(
        y=gender_names,
        x=gender_pcts[column_names[i]].tolist(),
        text=gender_counts[column_names[i]].tolist(),
        name=column_names[i],
        showlegend=False,
        orientation='h',
        marker=dict(color=colors[i])
    )
    gender_data.append(trace)

layout = go.Layout(
    barmode='stack',
    title="Percent funded (by Gender)"
)
fig = tools.make_subplots(rows=2, cols=1,
                          subplot_titles=["Percent funded (by Gender)", ""],
                          shared_xaxes=True,)

for x in overall_data:
    fig.append_trace(x, 1, 1)
for x in gender_data:
    fig.append_trace(x, 2, 1)

fig["layout"].update(barmode='stack')
fig["layout"]["yaxis2"].update(dict(domain=[0, 0.60]))
fig["layout"]["yaxis1"].update(dict(domain=[0.7, 1]))
fig['layout'].update(height=500, width=1000)


py.iplot(fig, filename='customizing-subplot-axes')
# Evaluate which countries are present in the loans DF
country_df = pd.DataFrame(data=loans["country"].value_counts()).reset_index()
country_df.columns = ["country", "count"]
# country_df.head()
# Evaluate which countries are available in the mpi_region_locations dataframe.

# There are a lot of rows which are filled with null values, so we'll drop any of those rows
mpi_region_locations = mpi_region_locations.drop(mpi_region_locations[mpi_region_locations.ISO.isnull()].index)
# Find the gap between countries present in the `loans` dataframe but missing from the MPL region locations dataframe.
missing = list()
for c in country_df.country.unique().tolist():
    if c not in mpi_region_locations.country.unique().tolist():
        missing.append(c)
# Add missing countries the the mpi_region_locations dataframe.
# This was done manually

missing = sorted(missing)
country_codes = ["BOL","CHL","COG","CRI","CIV","GEO","GUM","ISR","KSV","LBN","MDA","MMR","PSE","PAN","PRY","PRI",
                 "VCT","WSM","SLB","TZA","COD","TUR","USA","VNM","VGB"]
world_regions = ["Latin America and Caribbean","Latin America and Caribbean","Sub-Saharan Africa",
                 "Latin America and Caribbean","Sub-Saharan Africa","Europe and Central Asia",
                 "East Asia and the Pacific","Arab States","Europe and Central Asia","Arab States",
                 "Europe and Central Asia","South Asia","Arab States","Latin America and Caribbean",
                 "Latin America and Caribbean","Latin America and Caribbean","Latin America and Caribbean",
                 "East Asia and the Pacific","East Asia and the Pacific","Sub-Saharan Africa","Sub-Saharan Africa",
                 "Europe and Central Asia","North America","East Asia and the Pacific","Latin America and Caribbean"]

# Create a list of missing countries and their properties (fill_in_countries) to append to the existing mpl dataframe
fill_in_countries=[]
for i in range(len(country_codes)):
    countries = dict(
                    LocationName="N/A",
                    ISO=country_codes[i],
                    country=missing[i],
                    region="N/A",
                    world_region=world_regions[i],
                    MPI=0.0,
                    geo=(1000,1000),
                    lat=0,
                    lon=0,
                    )
    fill_in_countries.append(countries)
# Join the two dataframes so that all of the data is available in the `loans` dataframe.
mpi_region_locations = pd.concat([mpi_region_locations, pd.DataFrame(fill_in_countries)])

# Create a pivot table to keep only unique combinations of ISO, country, and world_region in the mpl dataframe
mpl_countries = mpi_region_locations.pivot_table(index=["ISO","country","world_region"]).reset_index()[["ISO","country","world_region"]]

# Merge the loans and new (complete) mpl_countries dataframes onto the loans dataframe.
# New columns in the loans dataframe will be: ISO, world_region.
loans = pd.merge(loans, mpl_countries, on="country", copy=False)
loans.head()
print("Columns in the dataset (Updated):\n", sorted(loans.columns.tolist()))
country_counts = loans.country.value_counts().sort_values(ascending=False).head(25)[::-1]

country_freq = loans.pivot_table(index=["country","ISO"], values="id", aggfunc=lambda x: len(x.unique())).reset_index()

tracebar = go.Bar(
    y=country_counts.index,
    x=country_counts.values,
    orientation = 'h',
    marker={
        "color":country_counts.values,
        "colorscale":"Viridis",
        "reversescale":True
    },
)
tracemap = dict(type="choropleth",
             locations=country_freq.ISO.tolist(),
             z=country_freq["id"].tolist(),
             text=country_freq.country.tolist(),
             colorscale="Viridis",
             reversescale=True,
             showscale=False
            )

data = [tracemap, tracebar]
layout = {
    "title": "Loans by Country",
    "height": 1000,
    "width": 1000,
      "geo": {
      "domain": {
          "x": [0, 1], 
          "y": [0.52, 1]
      }
    }
    ,
    "yaxis1": {
        "domain": [0, 0.5]
    },
    "xaxis1": {
        "domain": [0.1, 1]
    }
}
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='mapbar')
world_regions = pd.DataFrame(loans.world_region.value_counts()).reset_index()
world_regions.columns = ["world_region", "count"]
trace = go.Bar(x=world_regions["world_region"].tolist(),
              y=world_regions['count'].tolist())

names = world_regions["world_region"].tolist()
vals = world_regions["count"].tolist()

region_data = list()
for i in range(len(names)):
    trace = go.Bar(
        y=["World Regions"],
        x=[vals[i]],
        name=names[i],
        orientation='h',
    )
    region_data.append(trace)

layout = go.Layout(
    barmode='stack',
    title="Number of Loans by World Region",
    xaxis=dict(title="Number of Loans"),
    height=360, width=1000
)

fig = go.Figure(data=region_data, layout=layout)
py.iplot(fig)
world_region_sector_cts = loans.pivot_table(index="world_region", columns="sector", values="id", aggfunc=lambda x: len(x.unique()))

# Create pct dataframe
world_region_totals = np.sum(world_region_sector_cts, axis=1)
world_region_sector_pcts = world_region_sector_cts.copy()
for col in world_region_sector_cts.columns:
    world_region_sector_pcts[col] = round(world_region_sector_cts[col]/world_region_totals, 4)

world_region_sector_pcts = world_region_sector_pcts[["Food", "Agriculture", "Retail", "Services", "Education", 
                                                     "Clothing", "Housing", "Arts", "Transportation", "Health",
                                                     "Entertainment", "Personal Use", "Construction", "Manufacturing",
                                                    "Wholesale"]].sort_values(by="Food", ascending=True)
world_region_sector_cts = world_region_sector_cts[["Food", "Agriculture", "Retail", "Services", "Education", 
                                                     "Clothing", "Housing", "Arts", "Transportation", "Health",
                                                     "Entertainment", "Personal Use", "Construction", "Manufacturing",
                                                    "Wholesale"]].loc[world_region_sector_pcts.index]
sector_col_names = world_region_sector_pcts.columns
world_region_names = world_region_sector_pcts.index.tolist()


trace_list = list()
for i in range(len(sector_col_names)):
    trace = go.Bar(
        y=world_region_names,
        x=world_region_sector_pcts[sector_col_names[i]].tolist(),
        text=world_region_sector_cts[sector_col_names[i]].tolist(),
        name=sector_col_names[i],
#         showlegend=False,
        orientation='h',
    )
    trace_list.append(trace)

layout = go.Layout(
    barmode='stack',
    title="Sectors by World Region",
    yaxis=dict(tickangle=-55),
)
fig = go.Figure(data=trace_list, layout=layout)
py.iplot(fig)