import pandas as pd

import plotly.express as px

summary_df = pd.read_csv("../input/us-flights-with-coivid19-tsa-screening-officer/Summary_Stats_TSA_Positive.csv",index_col=0).T

summary_df["number_seats"] = summary_df["number_seats"] / 100

summary_df.columns = ["Number of Flights Departing", "Total Seats in Hundreds"]

formated_summary = summary_df.stack().reset_index()

formated_summary.columns = ["stage", "feature", "number"]

fig = px.funnel(formated_summary, x='number', y='stage', color='feature')

fig.show()


import pandas_profiling 

flights_df = pd.read_csv("../input/us-flights-with-coivid19-tsa-screening-officer/Flights_with_TSA_Contact.csv")

flights_df.profile_report(title='Profiling Report', html={'style':{'full_width':True}}, progress_bar=False, minimal=True)
!pip install -q ipyaggrid



from ipyaggrid import Grid



def simple_grid(df):



    column_defs = [{'headername':c,'field': c} for c in df.columns]



    grid_options = {

        'columnDefs' : column_defs,

        'enableSorting': True,

        'enableFilter': True,

        'enableColResize': True,

        'enableRangeSelection': True,

        'rowSelection': 'multiple',

    }



    g = Grid(grid_data=df,

             grid_options=grid_options,

             quick_filter=True,

             show_toggle_edit=True,

             sync_on_edit=True,

             export_csv=True,

             export_excel=True,

             theme='ag-theme-balham',

             show_toggle_delete=True,

             columns_fit='auto',

             index=False)

    return g
passenger_features = [("flight_departed","sum"), ("number_seats","sum"), ("load_factor", "mean"), ("weighted_seats", "sum")]

outbound_columns = ["Origin Airport", "Total Departed Flights", "Total Potential Passengers", "Average Load Factor", "Total Scaled Passengers", "Total Destination Airports"]

outbound_flights = flights_df.groupby("origin_airport").agg(["sum", "nunique", "mean"])[passenger_features]

outbound_flights.columns = outbound_flights.columns.get_level_values(0)

outbound_flights = outbound_flights.merge(flights_df.groupby("origin_airport").nunique()["dest_airport"], left_index=True, right_index=True).reset_index()

outbound_flights.columns = outbound_columns

outbound_flights["Total Potential Passengers"] = outbound_flights["Total Potential Passengers"].astype(int)

simple_grid(outbound_flights)
passenger_features = [("flight_departed","sum"), ("number_seats","sum"), ("load_factor", "mean"), ("weighted_seats", "sum")]

inbound_columns = ["Destination Airport", "Total Arrived Flights", "Total Potential Passengers", "Average Load Factor", "Total Scaled Passengers", "Total Origin Airports"]

inbound_flights = flights_df.groupby("dest_airport").agg(["sum", "nunique", "mean"])[passenger_features]

inbound_flights.columns = inbound_flights.columns.get_level_values(0)

inbound_flights = inbound_flights.merge(flights_df.groupby("dest_airport").nunique()["origin_airport"], left_index=True, right_index=True).reset_index()

inbound_flights.columns = inbound_columns

inbound_flights["Total Potential Passengers"] = inbound_flights["Total Potential Passengers"].astype(int)

simple_grid(inbound_flights)
passenger_features = [("flight_departed","sum"), ("number_seats","sum"), ("load_factor", "mean"), ("weighted_seats", "sum")]

route_columns = ["Origin Airport", "Destination Airport", "Total Arrived Flights", "Total Potential Passengers", "Average Load Factor", "Total Scaled Passengers"]

routes_flights = flights_df.groupby(["origin_airport", "dest_airport"]).agg(["sum", "nunique", "mean"])[passenger_features]

routes_flights.columns = routes_flights.columns.get_level_values(0)

routes_flights = routes_flights.reset_index()

routes_flights.columns = route_columns

routes_flights["Total Potential Passengers"] = routes_flights["Total Potential Passengers"].astype(int)

simple_grid(routes_flights)
import plotly.express as px

fig = px.bar(routes_flights, x="Total Scaled Passengers", y="Destination Airport", color='Origin Airport', orientation='h',

             hover_data=["Total Scaled Passengers", "Average Load Factor"],

             height=1600,

             title='Routes')

fig.show()