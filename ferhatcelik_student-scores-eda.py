import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class StudentEDA (object):

    def import_student_performance (self):
        df = pd.read_csv("../input/StudentsPerformance.csv")
        df.columns = [self.dataFrame_notation(x) for x in df.columns]
        self.data = df

        # level of education
        levels = {
            "master's degree" : 5,
            "bachelor's degree" : 4,
            "associate's degree": 3,
            "some college" : 2,
            "some high school" : 1,
            "high school" : 0
            }
        df["Level"] = df["ParentalLevelOfEducation"].apply(lambda x: levels[x])

        # all features
        print (df.info())
        for i in range(0, len(levels)):
            print ("\n\nLevel {} samples\n".format(i), df[df.Level == i].describe())
        return df

    def dataFrame_notation(self, x):
        str_parts = x.replace(" "," ").replace("/"," ").split()
        return "".join([s.capitalize() for s in str_parts])

    def explore_scores(self):
        df = self.data
        numerics = ["MathScore", "ReadingScore", "WritingScore"]
        colors = ["red", "green", "blue"]

        plt.title ("Relations on Scores")
        sns.heatmap(df[numerics].corr(), cmap='gist_heat', linewidth=5, annot=True, vmin=0.80, fmt="0.3f")
        plt.show()

        plt.title("student-index sorted by values")
        for i in range(len(numerics)):
            targetNumeric = numerics[i]
            the_color = colors[i]
            plt.subplot(3, 1, i+1)
            #plt.title(targetNumeric)
            plt.xlabel("student-index sorted by values")
            plt.ylabel(targetNumeric)
            values = list(df[targetNumeric].sort_values())
            plt.grid(linestyle=":")
            plt.plot(values, c=str(the_color), linewidth=1 ,alpha=1)
        plt.show()

        plt.title("histograms")
        bins = 30
        r = (0,100)
        for i in range(len(numerics)):
            targetNumeric = numerics[i]
            values = df[targetNumeric]
            plt.subplot(3, 1, i+1)
            plt.hist(values, histtype='bar', label=targetNumeric, alpha=0.2, color=colors[i], bins=bins, range=r)
            plt.hist(values, histtype='step',  color=colors[i], alpha=1, bins=bins, range=r)
            plt.grid(linestyle=":")
            plt.legend()
            plt.ylabel("frequency")
            plt.xlabel("score")

        plt.show()

    def explore_education_levels(self):
        df = self.data
        df["AvgScore"] = (df["MathScore"] + df["WritingScore"] + df["ReadingScore"]) / 3.0
        target_fields = ["Level", "AvgScore", "MathScore", "WritingScore", "ReadingScore"]
        df[target_fields].boxplot(by="Level", figsize=(12,12))
        plt.show()

    def explore_relation_scores_by_level (self, level = 0):
        df = self.data
        df = df[df.Level == level]
        plt.scatter(df["MathScore"], df["WritingScore"], color="red", label="Math vs Writing", alpha=0.5)
        plt.scatter(df["MathScore"], df["ReadingScore"], color="green", label="Math vs Reading", alpha=0.5)
        plt.title("LEVEL {} Score distributions".format(level))
        plt.grid(linestyle=":")
        plt.xlabel("MathScore")
        plt.ylabel("Score")

        plt.legend()
        plt.show()
eda = StudentEDA ()
df = eda.import_student_performance()
df.head()
eda.explore_scores()
numerics = ["MathScore", "ReadingScore", "WritingScore"]
print (df[numerics].info())
print (df[numerics].describe())

eda.explore_education_levels()
plt.show()
eda.explore_relation_scores_by_level(level=0)
eda.explore_relation_scores_by_level(level=3)
eda.explore_relation_scores_by_level(level=5)

