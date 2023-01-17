# General Data Science

import numpy as np

import pandas as pd

pd.set_option("display.max_columns", 50)



# Combinatorial Optimization

from ortools.linear_solver import pywraplp



# Miscellaneous

from pathlib import Path

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_path = Path("/kaggle/input/emoji-diet-nutritional-data-sr28/")

nutritional_information_path = data_path/"Emoji Diet Nutritional Data (g) - EmojiFoods (g).csv"

nutritional_requirements_path = data_path/"Min Dietary Reference Intake (DRI) - Sheet4.csv"
nutritional_information = pd.read_csv(nutritional_information_path)

nutritional_information
nutritional_information.set_index("name", inplace=True)

nutritional_information.drop("emoji", axis=1, inplace=True)

assert nutritional_information.isna().sum().sum() == 0, "Missing data."
nutritional_requirements = pd.read_csv(nutritional_requirements_path)

nutritional_requirements
nutritional_requirements = nutritional_requirements.T[0]

nutritional_requirements.name = "nutritional_requirements"

nutritional_requirements.fillna(0, inplace=True)

nutritional_requirements
solver = pywraplp.Solver(name="nutrition_solver", problem_type=pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

objective = solver.Objective()
items = {item: solver.NumVar(0., solver.infinity(), item) for item in nutritional_information.index}

for item in items.values():

    objective.SetCoefficient(item, 1)

objective.SetMinimization()
constraints = {item: solver.Constraint(requirement, solver.infinity()) for item, requirement in nutritional_requirements.iteritems()}

for item, item_info in nutritional_information.iterrows():

    for metric, value in item_info.iteritems():

        constraints[metric].SetCoefficient(items[item], value)
status = solver.Solve()

if status == solver.OPTIMAL:

    print("Found the optimal solution!")

elif status == solver.FEASIBLE:

    print("Found a feasible solution.")

else:

    print("Problem not solved.")
for item, value in items.items():

    solution = value.solution_value()

    if solution > 0.:

        print(item, "-", solution, "grams")