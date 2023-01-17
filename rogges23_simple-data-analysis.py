import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
movies = pd.read_csv("/kaggle/input/bollywood-box-office-20172020/bollywoodboxoffice_raw.csv")
movies.info()
movies.head()
movies.drop(["movie_url", "movie_director_url"], axis = 1, inplace = True)
movies[["movie_release_date", "movie_runtime"]] = movies["movie_movierelease"].str.split("|", expand = True)
movies.drop("movie_movierelease", axis = 1, inplace = True)
movies.columns = movies.columns.str.replace("^movie_", '')
movies.head()
revenues = ["opening", "weekend", "firstweek", "total", "total_worldwide"]
types = {col: 'float' for col in revenues}

def replacer(x):
    x = x.str.replace("^: ", "")
    x = x.str.replace("cr$", "")
    x = x.str.replace('^---', "0")
    return x.str.replace("\* $", "")

movies[revenues] = movies[revenues].apply(replacer)
movies[revenues] = movies[revenues].astype(types)

movies[revenues].head()
movies[["date", "year"]] = movies["release_date"].str.split(",", expand = True)
movies[["day", "month"]] = movies["date"].str.split(" ", expand = True)

movies.drop(["release_date", "date"], axis = 1, inplace = True)

movies.head()
time = movies["runtime"].str.split()

time = time.apply(lambda x: [int(x1) for x1 in x if x1 != 'hrs' and x1 != 'mins'])

def calc_runtime(x):
    if len(x) == 1:
        return 60 * x[0]
    else:
        return (60 * x[0]) + x[1]

movies["runtime"] = time.apply(calc_runtime)

movies.head()
movies.rename({"total": "domestic", "total_worldwide": "worldwide"}, axis = 1, inplace = True)
def cat_length(x):
    if x <= 120:
        return "Less than 2hrs"
    elif (x > 120) and (x < 180):
        return "Between 2 and 3 hours"
    elif x > 180:
        return "Longer than 3 hours"

movies["cat_runtime"] = movies["runtime"].apply(cat_length)
months = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
sns.set_style("darkgrid")
plt.style.use("fivethirtyeight")
plt.figure(figsize = (16, 8))
sns.boxplot(x = "month", y = "domestic", data = movies, order = months)
plt.ylabel("Crores INR")
plt.xlabel("Month")
plt.title("Domestic Gross of Indian Movies by Month")
sns.set_style("darkgrid")
plt.style.use("fivethirtyeight")
plt.figure(figsize = (16, 8))
sns.boxplot(x = "month", y = "worldwide", data = movies, order = months)
plt.ylabel("Crores INR")
plt.xlabel("Month")
plt.title("Worldwide Gross of Indian Movies by Month")
hundred = movies[movies["worldwide"] >= 100]
hundred.shape
plt.figure(figsize = (7, 5))
sns.set_style("darkgrid")
plt.style.use("tableau-colorblind10")
sns.countplot(x = "year", data = hundred)
plt.ylim(0, 25)
plt.figure(figsize = (7, 5))
sns.set_style("darkgrid")
plt.style.use("tableau-colorblind10")
sns.countplot(x = "cat_runtime", data = movies)
plt.ylim(0, 120)
plt.xlabel("Runtime of Movie")
opening_weekend = movies.sort_values(by = ["weekend"], ascending = False).head(10)
first_day = movies.sort_values(by = ["opening"], ascending = False).head(10)
sns.set_style("darkgrid")
plt.style.use("fivethirtyeight")
f, ax = plt.subplots(2, 1, figsize = (16, 14), squeeze = False)

sns.barplot(x = 'opening', y = 'name', data = first_day, ax = ax[0][0], palette = "inferno")
sns.barplot(x = 'weekend', y = 'name', data = opening_weekend, ax = ax[1][0], palette = 'inferno')

ax[0][0].set_ylabel('')
ax[0][0].set_xlabel('Crores INR')
ax[0][0].set_title('Biggest Opening Day')

ax[1][0].set_ylabel('')
ax[1][0].set_xlabel('Crores INR')
ax[1][0].set_title('Biggest Opening Weekend')