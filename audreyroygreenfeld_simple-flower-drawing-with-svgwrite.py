from math import cos, pi, sin, sqrt

from os import getcwd, system

from os.path import splitext

from time import sleep



import svgwrite





STROKE_WIDTH = 1

MULTIPLIER = 90

WIDTH = 8.5 * MULTIPLIER

HEIGHT = 11 * MULTIPLIER



PETALS = 6 # How many petals

R = 1 * MULTIPLIER



# Write to a SVG file with the same name as this script

# filename = '{}.svg'.format(splitext(__file__)[0])

filename = "six_petal_flower.svg"

dwg = svgwrite.Drawing(filename, size=(WIDTH, HEIGHT))



angle_step_size = 2 * pi / PETALS

phi = 0



while phi < 2 * pi:

    # Draw petal at current angle

    circle = dwg.circle(

        center=(WIDTH / 2 + R * cos(phi), HEIGHT / 2 + R * sin(phi)),

        r=0.7*MULTIPLIER,

        fill='#DEF7FF',

    )

    dwg.add(circle)

    circle2 = dwg.circle(

        center=(WIDTH / 2 + 2 * R * cos(phi), HEIGHT / 2 + 2 * R * sin(phi)),

        r=0.9*MULTIPLIER,

        fill='#DEF7FF',

    )

    dwg.add(circle2)

    phi += angle_step_size



# Top left corner

dwg.save()



from IPython.display import SVG, display

display(SVG(filename='six_petal_flower.svg'))