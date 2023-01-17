import pandas as pd
laptops = pd.read_csv("../input/laptops.csv", encoding="Latin-1")

laptops.info()
print(laptops.columns)
laptops_test = laptops.copy()
laptops_test.columns = ['A', 'B', 'C', 'D', 'E',
                        'G', 'F', 'H', 'I', 'J',
                        'K', 'L', 'M']
laptops_test.columns
def clean_col(col):
    col = col.strip()
    col = col.replace("Operating System", "os")
    col = col.replace(" ", "_")
    col = col.replace("(", "")
    col = col.replace(")", "")
    col = col.lower()
    return col
laptops.columns = [clean_col(c) for c in laptops.columns]
laptops.columns
print(laptops.iloc[:5,2:5])
print(laptops["screen_size"].dtype)
print(laptops["screen_size"].unique())
laptops["screen_size"] = laptops["screen_size"].str.replace('"', '')
print(laptops["screen_size"].unique())
laptops["screen_size"] = laptops["screen_size"].astype(float)
print(laptops["screen_size"].dtype)
print(laptops["screen_size"].unique())
laptops.rename({"screen_size": "screen_size_inches"}, axis=1, inplace=True)
laptops.columns
print(laptops.dtypes)
print(laptops["weight"].unique())
#laptops["weight"] = laptops["weight"].str.replace("kg", "").astype(float)
print(laptops.loc[laptops["weight"].str.contains('s'), "weight"])
laptops["weight"] = laptops["weight"].str.replace("kgs","").str.replace("kg","")
laptops["weight"] = laptops["weight"].astype(float)
laptops.rename({"weight": "weight_kg"}, axis=1, inplace=True)
laptops.rename({"weight": "weight_kg"}, axis=1, inplace=True)
laptops["price_euros"] = laptops["price_euros"].str.replace(",", ".").astype(float)
laptops.describe()
weight_describe = laptops["weight_kg"].describe()
print(weight_describe)
price_describe = laptops["price_euros"].describe()
print(price_describe)
laptops["gpu_manufacturer"] = (laptops["gpu"]
                                    .str.split(n=1,expand=True)
                                    .iloc[:,0]
                               )
laptops.head()
laptops["cpu_manufacturer"] = laptops["cpu"].str.split(n=1, expand=True).iloc[:,0]
laptops.head()
screen_res = laptops["screen"].str.rsplit(n=1, expand=True)
screen_res.columns = ["A", "B"]
screen_res.loc[screen_res["B"].isnull(), "B"] = screen_res["A"]
laptops["screen_resolution"] = (screen_res["B"]
                                    .str.split(n=1,expand=True)
                                    .iloc[:,0]
                                    )
laptops.head()
laptops["cpu"].unique()[:5]
laptops["cpu_speed_ghz"] = laptops["cpu"].str.replace("GHz", "").str.rsplit(n=1, expand=True).iloc[:,1].astype(float)
laptops.head(10)
mapping_dict = {
    'Android': 'Android',
    'Chrome OS': 'Chrome OS',
    'Linux': 'Linux',
    'Mac OS': 'macOS',
    'No OS': 'No OS',
    'Windows': 'Windows',
    'macOS': 'macOS'
}
laptops["os"] = laptops["os"].map(mapping_dict)
laptops.info()
print(laptops.isnull().sum())
print(laptops["os_version"].value_counts(dropna=False))
os_with_null_v = laptops.loc[laptops["os_version"].isnull(),"os"]
print(os_with_null_v.value_counts())
mac_os_versions = laptops.loc[laptops["os"] == "macOS", "os_version"]
print(mac_os_versions.value_counts(dropna=False))
laptops.loc[laptops["os"] == "macOS", "os_version"] = "X"
mac_os_versions = laptops.loc[laptops["os"] == "macOS", "os_version"]
print(mac_os_versions.value_counts(dropna=False))
print(laptops.loc[76:81, "storage"])
# replace 'TB' with 000 and rm 'GB'
laptops["storage"] = laptops["storage"].str.replace('GB','').str.replace('TB','000')

# split out into two columns for storage
laptops[["storage_1", "storage_2"]] = laptops["storage"].str.split("+", expand=True)

for s in ["storage_1", "storage_2"]:
    s_capacity = s + "_capacity_gb"
    s_type = s + "_type"
    # create new cols for capacity and type
    laptops[[s_capacity, s_type]] = laptops[s].str.split(n=1,expand=True)
    # make capacity numeric (can't be int because of missing values)
    laptops[s_capacity] = laptops[s_capacity].astype(float)
    # strip extra white space
    laptops[s_type] = laptops[s_type].str.strip()

# remove unneeded columns
laptops.drop(["storage", "storage_1", "storage_2"], axis=1, inplace=True)
laptops
laptops_dtypes = laptops.dtypes
cols = ['manufacturer', 'model_name', 'category', 'screen_size_inches',
        'screen', 'cpu', 'cpu_manufacturer', 'screen_resolution', 'cpu_speed_ghz', 'ram',
        'storage_1_type', 'storage_1_capacity_gb', 'storage_2_type',
        'storage_2_capacity_gb', 'gpu', 'gpu_manufacturer', 'os',
        'os_version', 'weight_kg', 'price_euros']
laptops = laptops[cols]

laptops.to_csv('laptops_cleaned.csv',index=False)
laptops_cleaned = pd.read_csv('laptops_cleaned.csv')
laptops_cleaned_dtypes = laptops_cleaned.dtypes
