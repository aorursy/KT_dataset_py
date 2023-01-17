import functools

import inspect

import itertools

import json

import os

import random



import numpy as np

import pandas as pd



from dataclasses import dataclass

from pathlib import Path

from skimage import transform

from skimage import util
def read_all_tasks(path):

    filenames = sorted(os.listdir(path))

    all_tasks = {}

    for task_file_name in filenames:

        full_path = str(Path.joinpath(path, task_file_name))



        with open(full_path, 'r') as file_obj:

            all_tasks[task_file_name] = json.load(file_obj)



    return all_tasks



data_path = Path("/kaggle/input/abstraction-and-reasoning-challenge/")
@dataclass

class Transformation:

    transform: object

    argument_ranges: dict





_grid_transformations = []





def grid_transformation(**argument_ranges):

    def decorator(func):

        _grid_transformations.append(

            Transformation(transform=func, argument_ranges=argument_ranges)

        )

        @functools.wraps(func)

        def wrapper(*args, **kwargs):

            return func(*args, **kwargs)

        return wrapper

    return decorator    
class Grid:

    def __init__(self, pixels):

        self.pixels = np.array(pixels)



    @property

    def shape(self):

        return self.pixels.shape

    

    @property

    def colors(self):

        return np.unique(self.pixels)

    

    def to_sub_grids(self):

        for color in self.colors:

            if not color:

                continue

            for mask in generate_object_masks(self.pixels == color):

                mask[mask > 0] = color

                yield Grid(mask.astype(self.pixels.dtype))



    def to_submission_output(self):

        str_pred = str([row.tolist() for row in self.pixels])

        str_pred = str_pred.replace(', ', '')

        str_pred = str_pred.replace('[[', '|')

        str_pred = str_pred.replace('][', '|')

        str_pred = str_pred.replace(']]', '|')

        return str_pred

    

    def __eq__(self, other):

        if not isinstance(other, Grid):

            return False

        

        return np.array_equal(self.pixels, other.pixels)



    def copy(self):

        return Grid(np.copy(self.pixels))



    @grid_transformation(size=lambda grid, target_grid, **_: [target_grid.shape] if grid.shape != target_grid.shape else [])

    def crop(self, *, size, **_):

        cropped = self.pixels[0:size[0], 0:size[1]]

        return Grid(cropped)



    @grid_transformation(scale=lambda grid, target_grid, **_: [tuple(a/b for a, b in zip(target_grid.shape, grid.shape))] if grid.shape != target_grid.shape else [])

    def rescale(self, *, scale, **_):

        scaled = transform.rescale(

            self.pixels,

            scale=scale,

            order=0,

            anti_aliasing=False,

            preserve_range=True,

        ).astype(self.pixels.dtype)

        return Grid(scaled)



    @grid_transformation(degrees=[90, 180, 270])

    def rotate(self, *, degrees, **_):

        rotated = transform.rotate(

            self.pixels,

            angle=degrees,

            order=0,

            resize=True,

            preserve_range=True,

        ).astype(self.pixels.dtype)

        return Grid(rotated)



    @grid_transformation(times=list(itertools.product((1, 2, 3), (1, 2, 3)))[1:])

    def repeat(self, *, times, **_):

        repeated = np.block([

            [np.copy(self.pixels) for _ in range(times[0])]

            for _ in range(times[1])

        ])

        return Grid(repeated)



    @grid_transformation(

        color_pair=lambda grid, target_grid, **_: list((c1, c2) for c1, c2 in itertools.product(grid.colors, target_grid.colors) if c1 != c2)

    )

    def change_color(self, *, color_pair, **_):

        colored = self.copy()

        colored.pixels[colored.pixels == color_pair[0]] = color_pair[1]

        return colored

        

    @classmethod

    def test(cls):

        test_submission_output = cls([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).to_submission_output()

        assert test_submission_output == "|123|456|789|", test_submission_output





Grid.test()
@dataclass

class Recipe:

    @dataclass

    class RecipeItem:

        transformation: Transformation

        arguments: dict = None

        skip: bool = False

            

    items: list



    def cook(self, grid, target_grid=None):

        for item in self.items:

            if item.skip:

                continue



            if item.arguments is None:

                assert target_grid is not None, "Unable to estimate arguments without target grid"

                argument_ranges = {

                    argument_name: (

                        argument_range(grid=grid, target_grid=target_grid)

                        if inspect.isfunction(argument_range)

                        else argument_range

                    )

                    for argument_name, argument_range in item.transformation.argument_ranges.items()

                }

                

                if all(len(argument_range) == 0 for argument_range in argument_ranges.values()):

                    item.skip = True

                    continue

                

                item.arguments = {

                    argument_name: random.choice(argument_range)

                    for argument_name, argument_range in argument_ranges.items()

                }



            grid = item.transformation.transform(grid, **item.arguments)

        

        return grid



    def __enter__(self):

        return self



    def __exit__(self, *_):

        self.items = [item for item in self.items if not item.skip]





class CookBook:

    def __init__(self):

        self._recipes = []

        self._best_recipe = None



    @property

    def latest_recipe(self):

        assert self._recipes, "No recipes created"

        return self._recipes[-1]



    @property

    def best_recipe(self):

        assert self._best_recipe is not None, "No recipe has been rated"

        return self._best_recipe[0]

    

    @property

    def solving_count(self):

        if self._best_recipe is None:

            return -1

        return self._best_recipe[1]



    def rate_latest_recipe(self, solving_count, err):

        is_the_new_best = (

            self._best_recipe is None or

            solving_count > self._best_recipe[1] or

            (

                solving_count == self._best_recipe[1] and err < self._best_recipe[2] or

                solving_count == self._best_recipe[1] and err == self._best_recipe[2] and len(self.latest_recipe.items) < len(self.best_recipe.items)

            )

        )



        if is_the_new_best:

            self._best_recipe = (self.latest_recipe, solving_count, err)



    def create_new_recipe(self):

        recipe = Recipe(

            items=[

                Recipe.RecipeItem(transformation)

                for transformation in random.choices(_grid_transformations, k=random.randint(1, len(_grid_transformations)))

            ]

        )



        self._recipes.append(recipe)



        return recipe
class Printer:

    _last_printed_line = None



    @classmethod

    def print_without_repeat(cls, msg):

        if msg == cls._last_printed_line:

            return

        print(msg)

        cls._last_printed_line = msg



    

def search_grid_transformations(tasks, max_iterations):

    transformations = {}



    for task_file_name, task in tasks.items():

        iteration = 0

        solved = False

        book = CookBook()

    

        while book.solving_count < len(task["train"]) and iteration < max_iterations:

            iteration += 1

            solving_count = 0

            err = 0



            with book.create_new_recipe() as recipe:

                for train_data in task["train"]:

                    input_grid = Grid(train_data["input"])

                    output_grid = Grid(train_data["output"])



                    try:

                        transformed_grid = recipe.cook(input_grid, output_grid)



                    except Exception as exception:

                        Printer.print_without_repeat(f"{task_file_name}: {exception}")

                        solving_count = -1

                        break



                    if transformed_grid.shape != output_grid.shape:

                        solving_count = -1

                        break

                    elif transformed_grid == output_grid:

                        solving_count += 1

                    else:

                        err += np.sum(np.abs(transformed_grid.pixels - output_grid.pixels))



            book.rate_latest_recipe(solving_count, err)

            solved = solving_count == len(task["train"])



        transformations[task_file_name] = {"recipe": book.best_recipe, "solved": book.solving_count == len(task["train"])}

    

    return transformations
def print_solved_problems(transformations):

    solved_problems = [t for t in transformations.items() if t[1]["solved"]]



    if not solved_problems:

        print("No problem solved")

        return



    for task_file_name, solver in solved_problems:

        print(task_file_name)

        for item in solver["recipe"].items:

            print(f"\t{item.transformation.transform.__name__} {item.arguments}")
training_path = data_path / "training"

training_tasks = read_all_tasks(training_path)
training_transformations = search_grid_transformations(training_tasks, max_iterations=1000)

print_solved_problems(training_transformations)
def generate_test_predictions(tasks, transformations):

    for task_file_name, task in tasks.items():

        if task_file_name not in transformations:

            print(f"No grid transformation estimate for {task_file_name}")

            continue



        transform = transformations[task_file_name]

        

        # There is no point to generate anything if even train data was not solved

        if not transform["solved"]:

            continue

        

        recipe = transform["recipe"]



        for idx, test_data in enumerate(task["test"]):

            output_id = f"{task_file_name.split('.')[0]}_{idx}"

            input_grid = Grid(test_data['input'])

            transformed_grid = recipe.cook(input_grid)

            yield output_id, transformed_grid.to_submission_output()
list(generate_test_predictions(training_tasks, training_transformations))
test_path = data_path / "test"

test_tasks = read_all_tasks(test_path)



submission = pd.read_csv(data_path / "sample_submission.csv", index_col="output_id")
test_transformations = search_grid_transformations(test_tasks, max_iterations=1000)

print_solved_problems(test_transformations)
for output_id, pred in generate_test_predictions(test_tasks, test_transformations):

    submission.loc[output_id, "output"] = pred



submission.to_csv("submission.csv")