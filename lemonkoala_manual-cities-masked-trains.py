import cv2
import os
import numpy  as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["figure.figsize"] = (30, 10)
tracks = pd.read_csv("../input/tracks.csv")
trains = pd.read_csv("../input/trains.csv", dtype={ "ConfigId": "str" })

pictures = {
    os.path.splitext(filename)[0]: cv2.imread(f"../input/pictures/pictures/{filename}")
    for filename in os.listdir("../input/pictures/pictures")
    if os.path.splitext(filename)[1] == ".jpg"
}
def show_image(image, dots=[]):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.scatter(
        x=[x for (x,y) in dots],
        y=[y for (x,y) in dots],
        color="orange"
    )
    plt.axis("off")
    plt.show()

show_image(pictures["03_A"])
trains[trains["ConfigId"] == "03"]
cities = {
    "Atlanta":         (2420, 785),
    "Boston":          (2525, 265),
    "Calgary":         ( 930,  315),
    "Charleston":      (2645,  775),
    "Chicago":         (2070,  530),
    "Dallas":          (1935, 1070),
    "Denver":          (1405,  790),
    "Duluth":          (1760,  455),
    "El Paso":         (1450, 1200),
    "Helena":          (1200,  515),
    "Houston":         (2075, 1150),
    "Kansas City":     (1815,  710),
    "Las Vegas":       ( 910, 1025),
    "Little Rock":     (2050,  865),
    "Los Angeles":     ( 710, 1195),
    "Miami":           (2905, 1090),
    "Montreal":        (2340,  200),
    "Nashville":       (2270,  730),
    "New Orleans":     (2300, 1085),
    "New York":        (2485,  385),
    "Oklahoma City":   (1820,  885),
    "Omaha":           (1740,  620),
    "Pittsburgh":      (2345,  475),
    "Phoenix":         (1085, 1155),
    "Portland":        ( 535,  555),
    "Raleigh":         (2515,  660),
    "Saint Louis":     (2020,  685),
    "Salt Lake City":  (1045,  765),
    "San Francisco":   ( 475,  965),
    "Santa Fe":        (1425,  990),
    "Sault St. Marie": (1995,  325),
    "Seattle":         ( 600,  460),
    "Toronto":         (2240,  340),
    "Vancouver":       ( 605,  365),
    "Washington":      (2585,  530),
    "Winnipeg":        (1450,  295),
}

show_image(pictures["03_A"], cities.values())
def mask_color(image, lower, upper):
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    
    mask = cv2.inRange(image, lower, upper)
    return cv2.bitwise_and(image, image, mask=mask)

mask_red  = mask_color(pictures["03_A"], [ 17,  15, 100], [ 50,  56, 200])
mask_blue = mask_color(pictures["03_A"], [100,  60,  15], [225, 135,  75])
show_image(mask_red, cities.values())
show_image(mask_blue, cities.values())
def points_in_line(c1, c2):    
    m = (c2[1] - c1[1]) / (c2[0] - c1[0])
    b =  c2[1] - c2[0] * m

    # go right to left if c1's x is greater
    step = 1 if c1[0] < c2[0] else -1

    return [
        (x, round(m * x + b))
        for x in range(c1[0], c2[0], step)
    ]
points_in_line(cities["Seattle"], cities["Vancouver"])
def is_color_in_points(mask, points):
    for point in points:
        index = point[::-1] # indexing the image is (y,x)
        color = mask[index]
        if tuple(color) != (0, 0, 0):
            return True

    return False
is_color_in_points(
    mask_blue,
    points_in_line(cities["Seattle"], cities["Vancouver"])
)
is_color_in_points(
    mask_blue,
    points_in_line(cities["Portland"], cities["Seattle"])
)
def detect_trains(mask):
    return set([
        (track["A"], track["B"])
        for index, track in tracks.iterrows()
        if is_color_in_points(
            mask,
            points_in_line(
                cities[track["A"]],
                cities[track["B"]]
            )
        )
    ])
detect_trains(mask_blue)
detect_trains(mask_red)