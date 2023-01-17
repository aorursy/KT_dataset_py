import matplotlib.pyplot as plt



import numpy as np

from numpy.random import seed



import pandas as pd



from scipy.stats import shapiro, ttest_1samp
df = pd.read_csv("../input/80-cereals/cereal.csv")
print(df.shape)
print(df.count())
print(df.isna().sum())
print(df.head())
df = df.replace(-1, np.nan)



print(df.isna().sum())
print(pd.isna(df).any(axis = 1).sum())
df = df.dropna()



print("Number of observations:", df.shape[0])
transformed_df = pd.DataFrame()

transformed_df["name"] = df["name"]



print(transformed_df)
roles_dict = {}
transformed_df["calorie_dist"] = np.abs(df["calories"] - 100)



print(transformed_df["calorie_dist"].describe())



# Show a histogram.

transformed_df.hist("calorie_dist")

plt.title("Fig. 1. Distances between Cereals' Calorie Values and the Recommended Amount 100 C")

plt.xlabel("Distance between Calorie Value and 100 C")

plt.ylabel("Frequency")

plt.show()
roles_dict["calorie_dist"] = "min"
transformed_df["fat"] = df["fat"] / 78 * 100



print(transformed_df["fat"].describe())



# Show a histogram.

transformed_df.hist("fat")

plt.title("Fig. 2. Percentages of the Recommended Daily Value of Fat Contained in Cereals")

plt.xlabel("% DV of Fat")

plt.ylabel("Frequency")

plt.show()
roles_dict["fat"] = "min"
transformed_df["fiber"] = df["fiber"] / 28 * 100



print(transformed_df["fiber"].describe())



# Show a histogram.

transformed_df.hist("fiber")

plt.title("Fig. 3. Percentages of the Recommended Daily Value of Dietary Fiber Contained in Cereals")

plt.xlabel("% DV of Dietary Fiber")

plt.ylabel("Frequency")

plt.show()
roles_dict["fiber"] = "max"
transformed_df["sugars"] = df["sugars"] / 50 * 100



print(transformed_df["sugars"].describe())



# Show a histogram.

transformed_df.hist("sugars")

plt.title("Fig. 4. Percentages of the Recommended Daily Value of Sugar Contained in Cereals")

plt.xlabel("% DV of Sugar")

plt.ylabel("Frequency")

plt.show()
print("Number of high-sugar cereals:", sum(transformed_df["sugars"] >= 20))
roles_dict["sugars"] = "min"
print(sum(df["carbo"] < df["fiber"]))
carb_cols = df[["fiber", "sugars", "carbo"]]



total_carb = pd.Series(

    carb_cols.sum(axis = 1),

    name = "total_carb",

)



print(carb_cols.join(total_carb))
transformed_df["total_carb"] = total_carb / 275 * 100



print(transformed_df["total_carb"].describe())



# Show a histogram.

transformed_df.hist("total_carb")

plt.title("Fig. 5. Percentages of the Recommended Daily Value of Total Carbohydrate Contained in Cereals")

plt.xlabel("% DV of Total Carbohydrate")

plt.ylabel("Frequency")

plt.show()
roles_dict["total_carb"] = "max"
transformed_df["protein"] = df["protein"] * 2



print(transformed_df["protein"].describe())



# Show a histogram.

transformed_df.hist("protein")

plt.title("Fig. 6. Percentages of the Recommended Daily Value of Protein Contained in Cereals")

plt.xlabel("% DV of Protein")

plt.ylabel("Frequency")

plt.show()
roles_dict["protein"] = "max"
print(df["vitamins"].value_counts())
transformed_df["vit_score"] = df["vitamins"].replace(

    [25, 0, 100],

    [3, 2, 1],

)



# Show a histogram.

transformed_df.hist("vit_score")

plt.title("Fig. 7. Cereals Scored by the Healthiness of the Amount of Vitamins Contained")

plt.xlabel("Scoring")

plt.ylabel("Frequency")

plt.show()
roles_dict["vit_score"] = "max"
transformed_df["potass"] = df["potass"] / 4700 * 100



print(transformed_df["potass"].describe())



# Show a histogram.

transformed_df.hist("potass")

plt.title("Fig. 8. Percentages of the Recommended Daily Value of Potassium Contained in Cereals")

plt.xlabel("% DV of Potassium")

plt.ylabel("Frequency")

plt.show()
roles_dict["potass"] = "max"
transformed_df["sodium"] = df["sodium"] / 2300 * 100



print(transformed_df["sodium"].describe())



# Show a histogram.

transformed_df.hist("sodium")

plt.title("Fig. 9. Percentages of the Recommended Daily Value of Sodium Contained in Cereals")

plt.xlabel("% DV of Sodium")

plt.ylabel("Frequency")

plt.show()
roles_dict["sodium"] = "min"
roles_df = pd.DataFrame({

    "feature": list(roles_dict.keys()),

    "role": list(roles_dict.values()),

})



print(roles_df)
x1 = np.linspace(0.0, 1, 1000)

x2 = -x1 + 1



plt.plot(x1, x2, label = "Frontier")

plt.scatter([0.1, 0.4, 0.9], [0.9, 0.4, 0.1], label = "Data Points")

plt.grid(True)



plt.title("Fig. 10. Example of Simple Additive Weighting")

plt.xlabel("Feature 1")

plt.ylabel("Feature 2")

plt.legend()



plt.axis("square")

plt.xlim(0, 1)

plt.ylim(0, 1)



plt.show()
x1 = np.linspace(0.1, 1, 1000)

x2 = 0.1 / x1



plt.plot(x1, x2, label = "Frontier")

plt.scatter([0.1, 0.4, 0.9], [0.9, 0.4, 0.1], label = "Data Points")

plt.grid(True)



plt.title("Fig. 11. Example of a Multiplicative Score Function")

plt.xlabel("Feature 1")

plt.ylabel("Feature 2")

plt.legend()



plt.axis("square")

plt.xlim(0, 1)

plt.ylim(0, 1)



plt.show()
print(roles_df)
# Add 0.01 to all values in order to avoid errors from 0's.



max_df = (df[["fiber", "protein", "potass"]] # Original data

             .join(transformed_df[["total_carb", "vit_score"]]) # Transformed data

             .add(0.01)

         )



min_df = (df[["fat", "sugars", "sodium"]] # Original data

             .join(transformed_df["calorie_dist"]) # Transformed data

             .add(0.01)

         )
# Get the product Series of each DataFrame.

max_prod = max_df.product(axis = 1)

min_prod = min_df.product(axis = 1)



# Divide `max_prod` by `min_prod` to get the scores.

score = max_prod.divide(min_prod)



# Append the final scores as a column to a new `leaderboard_df`.

leaderboard_df = pd.DataFrame()

leaderboard_df["name"] = transformed_df["name"]

leaderboard_df["score"] = score



print(leaderboard_df)
leaderboard_df["score_rank"] = leaderboard_df["score"].rank(

    method = "dense",

    ascending = False,

)



leaderboard_df = leaderboard_df.sort_values("score_rank")

print(leaderboard_df)
pd.set_option("display.max_rows", None)

print(leaderboard_df)
print(score.describe())



plt.hist(score)



plt.title("Fig. 12. Distribution of 74 Cereals' Raw Overall Healthiness Scores")

plt.xlabel("Healthiness Score")

plt.ylabel("Frequency")



plt.show()
# Seed the random number generator.

seed(1)



# Use the `scipy.stats.shapiro` function.

stat, p = shapiro(score)



print("Statistic:", stat)

print("p-value:", p)
# Get the top 3 scores.

top_3 = score.nlargest(3)



# Seed the random number generator.

seed(1)



# Calculate the T-test.

stat, p = ttest_1samp(top_3, score.mean())



print("Statistic:", stat)

print("p-value:", p)
# Get the bottom 3 scores.

bot_3 = score.nsmallest(3)



# Seed the random number generator.

seed(1)



# Calculate the T-test.

stat, p = ttest_1samp(bot_3, score.mean())



print("Statistic:", stat)

print("p-value:", p)