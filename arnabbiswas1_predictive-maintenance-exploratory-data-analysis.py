import os

import sys



import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt



matplotlib.style.use("Solarize_Light2")



%matplotlib inline
def check_null(df):

    """

    Returns percentage of rows containing missing data

    """

    return df.isna().sum() * 100/len(df)





def get_missing_dates(series, start_date, end_date, freq="D"):

    """

    Returns the dates which are missing in the series

    date_sr between the start_date and end_date

    

    series: Series consisting of date

    start_date: Start date in String format

    end_date: End date in String format

    """

    return pd.date_range(

        start=start_date, end=end_date, freq=freq).difference(series)





def check_duplicate(df, subset):

    """

    Returns if there are any duplicate rows in the DataFrame.

    

    df: DataFrame under consideration

    subset: Optional List of feature names based on which 

            duplicate rows are being identified. 

    """

    if subset is not None: 

        return df.duplicated(subset=subset, keep=False).sum()

    else:

        return df.duplicated(keep=False).sum()





def create_date_features(source_df, target_df, feature_name):

    '''

    Create new features related to dates

    

    source_df : DataFrame consisting of the timestamp related feature

    target_df : DataFrame where new features will be added

    feature_name : Name of the feature of date type which needs to be decomposed.

    '''

    target_df.loc[:, 'year'] = source_df.loc[:, feature_name].dt.year.astype('uint16')

    target_df.loc[:, 'month'] = source_df.loc[:, feature_name].dt.month.astype('uint8')

    target_df.loc[:, 'quarter'] = source_df.loc[:, feature_name].dt.quarter.astype('uint8')

    target_df.loc[:, 'weekofyear'] = source_df.loc[:, feature_name].dt.isocalendar().week.astype('uint8')

    

    target_df.loc[:, 'hour'] = source_df.loc[:, feature_name].dt.hour.astype('uint8')

    

    target_df.loc[:, 'day'] = source_df.loc[:, feature_name].dt.day.astype('uint8')

    target_df.loc[:, 'dayofweek'] = source_df.loc[:, feature_name].dt.dayofweek.astype('uint8')

    target_df.loc[:, 'dayofyear'] = source_df.loc[:, feature_name].dt.dayofyear.astype('uint8')

    target_df.loc[:, 'is_month_start'] = source_df.loc[:, feature_name].dt.is_month_start

    target_df.loc[:, 'is_month_end'] = source_df.loc[:, feature_name].dt.is_month_end

    target_df.loc[:, 'is_quarter_start']= source_df.loc[:, feature_name].dt.is_quarter_start

    target_df.loc[:, 'is_quarter_end'] = source_df.loc[:, feature_name].dt.is_quarter_end

    target_df.loc[:, 'is_year_start'] = source_df.loc[:, feature_name].dt.is_year_start

    target_df.loc[:, 'is_year_end'] = source_df.loc[:, feature_name].dt.is_year_end

    

    # This is of type object

    target_df.loc[:, 'month_year'] = source_df.loc[:, feature_name].dt.to_period('M')

    

    return target_df





def plot_boxh_groupby(df, feature_name, by):

    """

    Box plot with groupby

    

    df: DataFrame

    feature_name: Name of the feature to be plotted

    by: Name of the feature based on which groups are created

    """

    df.boxplot(column=feature_name, by=by, vert=False, 

                              figsize=(10, 6))

    plt.title(f'Distribution of {feature_name} by {by}')

    plt.show()

    



def plot_hist(df, feature_name, kind='hist', bins=100, log=True):

    """

    Plot histogram.

    

    df: DataFrame

    feature_name: Name of the feature to be plotted.

    """

    if log:

        df[feature_name].apply(np.log1p).plot(kind='hist', 

                                              bins=bins, 

                                              figsize=(15, 5), 

                                              title=f'Distribution of log1p[{feature_name}]')

    else:

        df[feature_name].plot(kind='hist', 

                              bins=bins, 

                              figsize=(15, 5), 

                              title=f'Distribution of {feature_name}')

    plt.show()





def plot_ts(series, figsize=(20, 6), title=None, xlabel="", ylabel=""):

    """

    Plot Time Series data. The series object should have date or time as index.

    

    series: Series object to be plotted.

    """

    series.plot(figsize=figsize, title=title)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.show()





def plot_barh(df, feature_name, normalize=True, 

              kind='barh', figsize=(15,5), sort_index=False, title=None):

    """

    Plot barh for a particular feature

    

    kind : Type of the plot

    

    """

    if sort_index==True:

        df[feature_name].value_counts(

                normalize=normalize, dropna=False).sort_index().plot(

                kind=kind, figsize=figsize, grid=True,

                title=title)

    else:   

        df[feature_name].value_counts(

                normalize=normalize, dropna=False).sort_values().plot(

                kind=kind, figsize=figsize, grid=True,

                title=title)

    

    plt.legend()

    plt.show()





def plot_boxh(df, feature_name, kind='box', log=True):

    """

    Box plot

    """

    if log:

        df[feature_name].apply(np.log1p).plot(kind='box', vert=False, 

                                                  figsize=(10, 6), 

                                                  title=f'Distribution of log1p[{feature_name}]')

    else:

        df[feature_name].plot(kind='box', vert=False, 

                              figsize=(10, 6), 

                              title=f'Distribution of {feature_name}')

    plt.show()

    



def plot_scatter(df, feature_x, feature_y, figsize=(10,10), 

                 title=None, xlabel=None, ylabel=None):

    """

    Plot satter     

    """

    df.plot.scatter(feature_x, feature_y, 

                    figsize=(8, 6), title=title, 

                    legend=None)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    plt.show()
# Read the data

DATA_DIR = "/kaggle/input/microsoft-azure-predictive-maintenance/"



telemetry_df = pd.read_csv(f"{DATA_DIR}/PdM_telemetry.csv")

errors_df = pd.read_csv(f"{DATA_DIR}/PdM_errors.csv")

maint_df = pd.read_csv(f"{DATA_DIR}/PdM_maint.csv")

failures_df = pd.read_csv(f"{DATA_DIR}/PdM_failures.csv")

machines_df = pd.read_csv(f"{DATA_DIR}/PdM_machines.csv")



# Format date & time. Sort based on date for better readability

tables = [telemetry_df, maint_df, failures_df, errors_df]

for df in tables:

    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S")

    df.sort_values(["datetime", "machineID"], inplace=True, ignore_index=True)

print(f"Shape of the Telemetry Records: {telemetry_df.shape}")

print("\n")

telemetry_df.head()
telemetry_df[telemetry_df.machineID == 1].head()
telemetry_df.machineID.nunique()
telemetry_df.datetime.describe(datetime_is_numeric=True)
get_missing_dates(telemetry_df.datetime, 

                  start_date="2015-01-01 06:00:00", 

                  end_date="2016-01-01 06:00:00", 

                  freq='H')
check_duplicate(telemetry_df, ['datetime', 'machineID'])
check_null(telemetry_df)
# Let's plot Vibrarion of Machine 1 for 2015

df_vib_machine_1 = telemetry_df[

    telemetry_df.machineID == 1][["datetime", "vibration"]].set_index("datetime")

plot_ts(df_vib_machine_1, title="Vibration of Machine 1", xlabel="Time", ylabel="Vibration")
# Let's plot voltage of Machine 2 for 1st two weeks of 2015

df_vib_machine_1 = telemetry_df[

    (telemetry_df.machineID == 2) & (

        telemetry_df.datetime.dt.isocalendar().week.isin(

            [1, 2, 3]))][["datetime", "volt"]].set_index("datetime")

plot_ts(df_vib_machine_1, title="Volatage of Machine 2", xlabel="Time", ylabel="Voltage")
telemetry_df = create_date_features(telemetry_df, telemetry_df, "datetime")

telemetry_df.head()
plot_boxh_groupby(telemetry_df, feature_name="volt", by="month_year")
plot_boxh_groupby(telemetry_df[telemetry_df.machineID == 80], feature_name="volt", by="month_year")
plot_hist(telemetry_df, feature_name="volt", log=False, bins=1000)
for name in ['rotate', 'pressure', 'vibration']:

    plot_hist(telemetry_df, feature_name=name, log=False, bins=1000)
print(f"Shape of the Error Records: {errors_df.shape}")

print("\n")

errors_df.head()
# Sort the Error based "datetime", "machineID", "errorID" for better readability

errors_df = errors_df.sort_values(["datetime", "machineID", "errorID"]).reset_index(drop=True)
errors_df.machineID = errors_df.machineID.astype('category')

errors_df.errorID = errors_df.errorID.astype('category')
errors_df.datetime.describe(datetime_is_numeric=True)
check_duplicate(errors_df, ['datetime', 'machineID', 'errorID'])
check_null(errors_df)
plot_barh(errors_df, 

          feature_name="errorID", 

          figsize=(10, 6), 

          normalize=False,

          title="Different Types of Errors")
plot_barh(errors_df, "machineID", figsize=(6, 20), normalize=False, title="Number of errors across MachineID")
df_errors = errors_df.groupby(["machineID", "errorID"]).size().reset_index()

df_errors.columns = ["machineID", "errorID", "errorValues"]

#df_errors_pivot = pd.pivot(df_errors, index="machineID", columns="errorID", values="errorValues").reset_index().rename_axis(None, axis=1)

df_errors_pivot = pd.pivot(df_errors, index="machineID", columns="errorID", values="errorValues").rename_axis(None, axis=1)



df_errors_pivot.plot.bar(stacked=True, figsize=(20, 6), title="Count of Errors for different Machines")

plt.xlabel("Machine ID")

plt.ylabel("Number of Errors");
plot_ts(

    errors_df.datetime.dt.date.value_counts().sort_index(), 

    figsize=(20, 6), 

    title="Number of Errors Across Days", 

    xlabel="Time",

    ylabel="Number of Errors")
df_temp = errors_df[errors_df.machineID.isin([1, 2])].datetime.dt.date.value_counts().sort_index()

df_temp.plot(style="k.", figsize=(8, 4), title="Number of Errors Across Days for Machine 1 & 2")

plt.ylabel("Count of Errors")

plt.xlabel("Time")

plt.show()
errors_df['date'] = errors_df.datetime.dt.date



errors_df.groupby('date').size().hist(bins=20, figsize=(10, 6))

plt.title("Distribution of Number of Errors Per Day")

plt.xlabel("Number of Errors on a Particular Day")

plt.ylabel("Frequency")

plt.show()
print(f"Shape of the Maintenance Records: {maint_df.shape}")

print("\n")

maint_df.head()
for name in ["machineID", "comp"]:

    maint_df[name] = maint_df[name].astype("category")

    

maint_df.sort_values(["datetime", "machineID", "comp"], inplace=True)



# Add date related features.

maint_df = create_date_features(maint_df, maint_df, "datetime")
maint_df.datetime.describe(datetime_is_numeric=True)
get_missing_dates(maint_df.datetime, 

                  start_date="2014-06-01 06:00:00", 

                  end_date="2016-01-01 06:00:00", 

                  freq='H')
plot_barh(maint_df, "month_year", normalize=False, sort_index=True)
plot_barh(maint_df, 

          feature_name="comp", 

          figsize=(10, 6), 

          normalize=False,

          title="Components Replaced")
plot_barh(maint_df, "machineID", 

          figsize=(6, 20), 

          normalize=False, 

          title="Number of Maintenance Records across MachineID", 

          sort_index=True)
df_maint = maint_df.groupby(["machineID", "comp"]).size().reset_index()

df_maint.columns = ["machineID", "comp", "num_comp"]

df_maint_pivot = pd.pivot(df_maint, index="machineID", columns="comp", values="num_comp").rename_axis(None, axis=1)



df_maint_pivot.plot.bar(stacked=True, figsize=(20, 6), title="Count of Components Replaced for different Machines")

plt.xlabel("Machine ID")

plt.ylabel("Number of Components Replaced");
maint_df.datetime.dt.date.value_counts().plot(

    style="k.", 

    figsize=(20, 4), 

    title="Number of Maintenance Records Across Time")

plt.ylabel("Number of Maintenance Records")

plt.xlabel("Time")

plt.show()
print(f"Shape of the Machines Data: {machines_df.shape}")

print("\n")

machines_df.head()
plot_boxh(machines_df, feature_name="age", log=False)
# Create a DF with number of errors, maintenance records and failure records across machines



# Create a DF consisting of number of erros across Machines

erros_across_machine = errors_df.groupby("machineID").size()

erros_across_machine = pd.DataFrame(erros_across_machine, columns=["num_errors"]).reset_index()



machines_errors_df = pd.merge(machines_df, erros_across_machine, how='left', on="machineID")



# Create a DF consisting of number of maintenance records across Machines

maint_across_machine = maint_df.groupby("machineID").size()

maint_across_machine = pd.DataFrame(maint_across_machine, columns=["num_maint"]).reset_index()



machines_errors_df = pd.merge(machines_errors_df, maint_across_machine, how='left', on="machineID")



# Create a DF consisting of number of failure records across Machines

failure_across_machine = failures_df.groupby("machineID").size()

failure_across_machine = pd.DataFrame(failure_across_machine, columns=["num_failure"]).reset_index()



machines_errors_df = pd.merge(machines_errors_df, failure_across_machine, how='left', on="machineID")



machines_errors_df.head()
plot_scatter(machines_errors_df, "age", "num_errors", 

             title="Age vs Number of Errors", 

             xlabel="Age", ylabel="Number of Errors")
plot_scatter(machines_errors_df, "age", "num_maint", 

             title="Age vs Number of Maintenance Records", 

             xlabel="Age", ylabel="Number of Maintenance Records")
plot_scatter(machines_errors_df, "age", "num_failure", 

             title="Age vs Number of Failure Records", 

             xlabel="Age", ylabel="Number of Failure Records")
machines_errors_df.corr()