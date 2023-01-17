# Function: enrich_dataset_with_time(enr_dataset, enr_date_field, *arg)
#   Add following new features to the input dataframe
#   - date_new: input date is written as pandas date
#   - year: only year extracted from date
#   - month: only month extracted from date
#   - day: only day extracted from date
#   - week_day: 1:monday, 2:tuesday, 3:wednesday, ... , 6:saturday, 7:sunday
#   - week_of_year: from 1 to 52
#   - is_weekend: 1:weekend, 0:not weekend
#   - is_working_day: 1:working day, 0:not working day (it means weekend or holiday)
#   - is_holiday: 1:holiday, 0:not holiday
#   - is_first_work_day: 1:first working day of the month, 0:not first working day
#   - is_last_work_day: 1:last working day of the month, 0:not last working day
# Input: 
#   - pandas dataframe (enr_dataset)
#   - name of the column which has date (enr_date_field), source of all new features
#   - list (enr_holidays), optional, if you feed it working days will be updated and is_holiday feature will be added
# Output:
#   - pandas dataframe, the same dataset with new columns
# Sample Calling:
#   holidays = ["2011-1-1", "2011-1-4", "2011-1-6", "2013-7-6", "2012-1-2"]
#   enrich_dataset_with_time(dataset, "date_column", holidays)

def enrich_dataset_with_time(enr_dataset, enr_date_field, *arg):
    
    d = pd.DatetimeIndex(enr_dataset[enr_date_field])
    c = "date_new"
    
    enr_dataset[c] = d.date
    enr_dataset["year"] = d.year
    enr_dataset["month"] = d.month
    enr_dataset["day"] = d.day
    enr_dataset["week_day"] = d.weekday + 1
    enr_dataset["week_of_year"] = d.weekofyear
    enr_dataset["is_weekend"] = 0
    enr_dataset["is_working_day"] = 1
    enr_dataset.loc[enr_dataset["week_day"] == 6, "is_weekend"] = 1
    enr_dataset.loc[enr_dataset["week_day"] == 7, "is_weekend"] = 1
    enr_dataset.loc[enr_dataset["week_day"] == 6, "is_working_day"] = 0
    enr_dataset.loc[enr_dataset["week_day"] == 7, "is_working_day"] = 0

    if len(arg) == 1:
        enr_holidays = pd.DataFrame(arg[0], columns=[c])
        enr_holidays[c] = pd.DatetimeIndex(enr_holidays[c]).date
        enr_holidays["is_holiday"] = 1
        enr_dataset = pd.merge(enr_dataset, enr_holidays, how='left', on=[c, c])
        enr_dataset["is_holiday"].fillna(0, inplace=True)
        enr_dataset["is_holiday"] = enr_dataset["is_holiday"].astype("int")
        enr_dataset.loc[enr_dataset["is_holiday"] == 1, "is_working_day"] = 0

    temp = enr_dataset[enr_dataset["is_working_day"] == 1]    
    temp = pd.DataFrame(temp.groupby(["year", "month"]).agg("min").reset_index()[c])
    temp["is_first_work_day"] = 1
    enr_dataset = pd.merge(enr_dataset, temp, how='left', on=[c, c])
    enr_dataset["is_first_work_day"].fillna(0, inplace=True)
    enr_dataset["is_first_work_day"] = enr_dataset["is_first_work_day"].astype("int")

    temp = enr_dataset[enr_dataset["is_working_day"] == 1]    
    temp = pd.DataFrame(temp.groupby(["year", "month"]).agg("max").reset_index()[c])
    temp["is_last_work_day"] = 1
    enr_dataset = pd.merge(enr_dataset, temp, how='left', on=[c, c])
    enr_dataset["is_last_work_day"].fillna(0, inplace=True)
    enr_dataset["is_last_work_day"] = enr_dataset["is_last_work_day"].astype("int")

    return enr_dataset
    
import pandas as pd
dataset = pd.read_csv('../input/dataset_time_series.csv')
dataset.head()
a = enrich_dataset_with_time(dataset, "date_column")
a.head()
holidays = ["2011-1-1", "2011-1-4", "2011-1-6", "2013-7-6", "2012-1-2"]
b = enrich_dataset_with_time(dataset, "date_column", holidays)
b.head()