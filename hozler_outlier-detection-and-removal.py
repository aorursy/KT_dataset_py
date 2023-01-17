import pandas as pd
df_train = pd.read_csv("../input/train.csv")
df_train.head()
# Function: print_quantile_info(qu_dataset, qu_field)
#   Print out the following information about the data
#   - interquartile range
#   - upper_inner_fence
#   - lower_inner_fence
#   - upper_outer_fence
#   - lower_outer_fence
#   - percentage of records out of inner fences
#   - percentage of records out of outer fences
# Input: 
#   - pandas dataframe (qu_dataset)
#   - name of the column to analyze (qu_field)
# Output:
#   None

def print_quantile_info(qu_dataset, qu_field):
    a = qu_dataset[qu_field].describe()
    
    iqr = a["75%"] - a["25%"]
    print("interquartile range:", iqr)
    
    upper_inner_fence = a["75%"] + 1.5 * iqr
    lower_inner_fence = a["25%"] - 1.5 * iqr
    print("upper_inner_fence:", upper_inner_fence)
    print("lower_inner_fence:", lower_inner_fence)
    
    upper_outer_fence = a["75%"] + 3 * iqr
    lower_outer_fence = a["25%"] - 3 * iqr
    print("upper_outer_fence:", upper_outer_fence)
    print("lower_outer_fence:", lower_outer_fence)
    
    count_over_upper = len(qu_dataset[qu_dataset[qu_field]>upper_inner_fence])
    count_under_lower = len(qu_dataset[qu_dataset[qu_field]<lower_inner_fence])
    percentage = 100 * (count_under_lower + count_over_upper) / a["count"]
    print("percentage of records out of inner fences: %.2f"% (percentage))
    
    count_over_upper = len(qu_dataset[qu_dataset[qu_field]>upper_outer_fence])
    count_under_lower = len(qu_dataset[qu_dataset[qu_field]<lower_outer_fence])
    percentage = 100 * (count_under_lower + count_over_upper) / a["count"]
    print("percentage of records out of outer fences: %.2f"% (percentage))
print_quantile_info(df_train, "target")
# Function: remove_outliers_using_quantiles(qu_dataset, qu_field, qu_fence)
#   1- Remove outliers according to the given fence value and return new dataframe.
#   2- Print out the following information about the data
#      - interquartile range
#      - upper_inner_fence
#      - lower_inner_fence
#      - upper_outer_fence
#      - lower_outer_fence
#      - percentage of records out of inner fences
#      - percentage of records out of outer fences
# Input: 
#   - pandas dataframe (qu_dataset)
#   - name of the column to analyze (qu_field)
#   - inner (1.5*iqr) or outer (3.0*iqr) (qu_fence) values: "inner" or "outer"
# Output:
#   - new pandas dataframe (output_dataset)

def remove_outliers_using_quantiles(qu_dataset, qu_field, qu_fence):
    a = qu_dataset[qu_field].describe()
    
    iqr = a["75%"] - a["25%"]
    print("interquartile range:", iqr)
    
    upper_inner_fence = a["75%"] + 1.5 * iqr
    lower_inner_fence = a["25%"] - 1.5 * iqr
    print("upper_inner_fence:", upper_inner_fence)
    print("lower_inner_fence:", lower_inner_fence)
    
    upper_outer_fence = a["75%"] + 3 * iqr
    lower_outer_fence = a["25%"] - 3 * iqr
    print("upper_outer_fence:", upper_outer_fence)
    print("lower_outer_fence:", lower_outer_fence)
    
    count_over_upper = len(qu_dataset[qu_dataset[qu_field]>upper_inner_fence])
    count_under_lower = len(qu_dataset[qu_dataset[qu_field]<lower_inner_fence])
    percentage = 100 * (count_under_lower + count_over_upper) / a["count"]
    print("percentage of records out of inner fences: %.2f"% (percentage))
    
    count_over_upper = len(qu_dataset[qu_dataset[qu_field]>upper_outer_fence])
    count_under_lower = len(qu_dataset[qu_dataset[qu_field]<lower_outer_fence])
    percentage = 100 * (count_under_lower + count_over_upper) / a["count"]
    print("percentage of records out of outer fences: %.2f"% (percentage))
    
    if qu_fence == "inner":
        output_dataset = qu_dataset[qu_dataset[qu_field]<=upper_inner_fence]
        output_dataset = output_dataset[output_dataset[qu_field]>=lower_inner_fence]
    elif qu_fence == "outer":
        output_dataset = qu_dataset[qu_dataset[qu_field]<=upper_outer_fence]
        output_dataset = output_dataset[output_dataset[qu_field]>=lower_outer_fence]
    else:
        output_dataset = qu_dataset
    
    print("length of input dataframe:", len(qu_dataset))
    print("length of new dataframe after outlier removal:", len(output_dataset))
    
    return output_dataset
df_train.dropna(inplace=True)
new_dataset = remove_outliers_using_quantiles(df_train, "target", "inner")
new_dataset.head()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
sns.distplot(df_train["target"]);
plt.title('Histogram for Customer Loyalty Before Removal')
plt.figure(figsize=(15,10))
sns.distplot(new_dataset["target"]);
plt.title('Histogram for Customer Loyalty After Removal')