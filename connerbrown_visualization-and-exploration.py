import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import scipy.stats as sp
import scipy.optimize as spo
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

data_path = '../input/data.csv'
df = pd.read_csv(data_path)
df.drop('URL', axis = 1, inplace = True)
num_rows = len(df.index)
print ("Number of rows\t{}".format(num_rows))
df.head()
print ("Range of Position: {} - {}".format(df.Position.min(), df.Position.max()))
print ("Range of Streams: {} - {}".format(df.Streams.min(), df.Streams.max()))
print (df.loc[df.Streams.idxmax()])
df_nans = df.isnull()
print (df_nans.sum())
# find percent similar NaN rows between Artist and Track Name
track_name_nans = df_nans['Track Name'][df_nans['Track Name'] == True].index
artist_nans = df_nans['Artist'][df_nans['Artist'] == True].index
nans_overlap = (track_name_nans == artist_nans).sum() / df_nans['Track Name'].sum() * 100.0
print ("\nPercent Overlap: {}%".format(nans_overlap))
df_eng = df.copy()
df_eng.drop(track_name_nans, inplace = True)
print ("First Date: {}\nLast Date: {}".format(df_eng.Date.min(),df_eng.Date.max())) # check first and last dates
Dates = pd.to_datetime(df_eng.Date) # redundant date format check
Days = Dates.sub(Dates[0], axis = 0) # subtract dates in pandas
Days = Days / np.timedelta64(1, 'D') # convert to float
print ("First Day Elapsed: {}\nLast Day Elapsed: {}".format(Days.min(), Days.max())) # check converted first and last days elapsed
df_eng['Days'] = Days # add column to our dataframe
df_eng.drop('Date', axis = 1, inplace = True)
region_dict = {'ar':'Argentina', 'at':'Austria', 'au':'Australia', 'be':'Belgium', 'bo':'Bolivia', 'br':'Brazil', 'ca':'Canada', 'ch':'Switzerland', 'cl':'Chile', 'co':'Columbia', 'cr':'CostaRica', 'cz':'CzechRepublic', 'de':'Germany', 'dk':'Denmark', 'do':'DominicanRepublic',
 'ec':'Ecuador', 'ee':'Estonia', 'es':'Spain', 'fi':'Finland', 'fr':'France', 'gb':'UnitedKingdom', 'global':'World', 'gr':'Greece', 'gt':'Guatemala', 'hk':'HongKong', 'hn':'Honduras', 'hu':'Hungary', 'id':'Indonesia', 'ie':'Ireland',
 'is':'Iceland', 'it':'Italy', 'jp':'Japan', 'lt':'Lithuania', 'lu':'Luxemborg', 'lv':'Latvia', 'mx':'Mexico', 'my':'Malaysia', 'nl':'Netherlands', 'no':'Norway', 'nz':'NewZealand', 'pa':'Panama', 'pe':'Peru', 'ph':'Philippines', 'pl':'Poland',
 'pt':'Portugal', 'py':'Paraguay', 'se':'Sweden', 'sg':'Singapore', 'sk':'Slovakia', 'sv':'ElSalvador', 'tr':'Turkey', 'tw':'Taiwan', 'us':'USA', 'uy':'Uruguay',}
# sort by Track Name, then Region, then Days
df_eng.sort_values(['Track Name','Region','Days'], inplace = True)
DRank = np.zeros(len(df_eng.index)-1)

# create a current/previous df: current starts second row, previous ends second to last row
df_eng_curr = df_eng[1:].reset_index(drop = True)
df_eng_prev = df_eng[:-1].reset_index(drop = True)

# matrix operations: matrix-compare current and previous (shifted by 1) for columns: Track Name, Artist, Region 
next_position = (df_eng_prev == df_eng_curr)
# also check difference: current day - prev day == 1.
next_day = df_eng_curr.Days.sub(df_eng_prev.Days)
next_position.Days = (next_day == pd.Series(np.ones(len(next_day.index))))
# Get boolean array of all four. True -> current position - prev position. False -> 201 - current position
consecutive_positions = next_position['Track Name'] & next_position.Artist & next_position.Region & next_position.Days
# arrays of indices and values for true and false
true_subtract = df_eng_curr.Position[consecutive_positions] - df_eng_prev.Position[consecutive_positions]
false_subtract = 200 - df_eng_curr.Position[~consecutive_positions]
# set DRank values of true and false at true and false indices
DRank[true_subtract.index] = true_subtract
DRank[false_subtract.index] = false_subtract
# handle first row
DRank = np.insert(DRank,0,201 - df_eng_prev['Position'][0])
print (DRank[len(df_eng.index)-5:])
print (df_eng[len(df_eng.index)-5:])
# put DRank into the df_eng
df_eng['DRank'] = DRank
# correlation/Linear regression: Streams and Position
df_streams_position = df_eng.groupby(['Position']).sum()
df_streams_position.reset_index(inplace = True)
float_streams = df_streams_position['Streams'].astype(np.float)
# linear regression
slope_lin, intercept_lin, _,_,_ = sp.linregress(df_streams_position['Position'],float_streams)
regress_length = int(df_streams_position['Position'].max())
regress_lin = range(regress_length) * (np.ones(regress_length) * float(slope_lin)) + np.ones(regress_length) * float(intercept_lin)
# exponential regression
def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c
exp_fit, _ = spo.curve_fit(exp_func,  df_streams_position['Position'],  float_streams, p0=[-29000000000.0,1.064250352,4617322502.0])
exp_y = exp_func(df_streams_position['Position'], *exp_fit)
# correlation
corr_streams_position = float_streams.corr(df_streams_position['Position'])
print ("\t\t\tCorrelation\nPosition/Streams:\t{}".format(round(corr_streams_position,3)))
corr_lin = float_streams.corr(pd.Series(regress_lin))
print ("Linear Regression:\t{}".format(round(corr_lin,3)))
corr_exp = float_streams.corr(exp_y)
print ("Exponential Fit:\t{}".format(round(corr_exp,3)))
# plot Position vs Streams
fig_streams_position, ax_streams_position = plt.subplots()
ax_streams_position.plot(df_streams_position['Position'],float_streams);
ax_streams_position.plot(regress_lin,'r')
ax_streams_position.plot(df_streams_position['Position'], exp_y,'k')
ax_streams_position.set_title('Position vs Streams');
# Streams by region
df_streams = df_eng.groupby(['Region']).sum()
df_streams_sorted = df_streams.sort_values(by = ['Streams'], ascending=False)
top_10 = df_streams_sorted[1:11]
fig_streams, ax_streams = plt.subplots()
ax_streams.bar(range(len(top_10.index)), top_10.Streams, width = 1);
ax_streams.set_xticks(range(len(top_10.index)))
ax_streams.set_xticklabels(top_10.index);
ax_streams.set_yticks([]);
ax_streams.set_title('Billions of Streams by Region')
# display number above each bar
top_10_array = np.array(top_10['Streams'])
for x,y in zip(range(len(top_10_array)),top_10_array):
    ax_streams.text(x,y,round(y / 1000000000,1),horizontalalignment='center');
# top 10 regions - stream time series
df_stream_time = df_eng.groupby(['Region','Days']).sum()
df_stream_time.reset_index(inplace = True)
top_10_time = df_stream_time[df_stream_time['Region'].isin(top_10.index)]
plt.figure(figsize = (18,10))
for region in top_10.index:
    temp_df = top_10_time[top_10_time['Region'] == region]
    plt.plot(temp_df['Days'], temp_df['Streams'], label = region_dict[region]);
# Weekly peaks? Fridays are -2,5,12,19
fridays = list(range(0,int(top_10_time['Days'].max()),7))
fridays -= np.ones(len(fridays)) * 2
for friday in fridays[:6]:
    sat_plot = plt.axvline(friday, linestyle = '--')
sat_plot.set_label('Fridays')
# label seasons
plt.legend();
# 4 US surges (78, 103, 236, 358)
df_us_stream_time = df_stream_time[df_stream_time['Region'] == 'us'].reset_index()
# streams surge locator function
def surge_locator(df, lower, upper):
    surge_idx = df['Streams'][lower:upper].idxmax() # look between lower and upper
    surge_stream_time = df.loc[surge_idx]
    surge = surge_stream_time['Days']
    df_surge = df_eng[df_eng['Region'] == 'us']
    df_surge = df_surge[df_surge['Days'] == surge]
    return (df_surge.sort_values(by = ['Position'])[:10])

# first surge = 78, 96540599
first_surge = surge_locator(df_us_stream_time, 60, 90)
print (first_surge)
# second surge = 103, 103538141
second_surge = surge_locator(df_us_stream_time, 90, 120)
print (second_surge)
# third surge = 236, 89480323
third_surge = surge_locator(df_us_stream_time, 210, 250)
print (third_surge)
# fourth surge = 358, 92597380
fourth_surge = surge_locator(df_us_stream_time, 330, 370)
print (fourth_surge)
