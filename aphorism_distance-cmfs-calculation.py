!pip install colour-science
import colour

import numpy

import matplotlib
LMS_CMFS = colour.CMFS["Stockman & Sharpe 2 Degree Cone Fundamentals"]

CMFS = colour.CMFS["CIE 2012 2 Degree Standard Observer"]
def calculate_length(coordinates):

    # Calculate the total distance as given by the sum of

    # lengths between coordinates. This is synonymous

    # with the function below using Numpy's diff and

    # norm approach.

    difference = numpy.diff(coordinates, axis=0)

    squared = numpy.power(difference, 2.0)

    sum_squared = numpy.sum(squared, axis=1)

    distance = numpy.power(sum_squared, (1.0 / 2.0))

    total = numpy.sum(distance)

    return total



def calculate_distance(coordinates):

    return numpy.sum(

        numpy.linalg.norm(

            numpy.diff(coordinates, axis=0),

            axis=1

        )

    )
CMFS_xy = colour.XYZ_to_xy(CMFS.values)



# Calculate the distance of the horseshoe without

# the magenta axis.

spectral_locus_open = calculate_distance(CMFS_xy)



CMFS_xy_closed = numpy.concatenate(

    (CMFS_xy, numpy.reshape(CMFS_xy[0], (1, 2)))

)



spectral_locus_closed = calculate_distance(CMFS_xy_closed)



magenta_length = spectral_locus_closed - spectral_locus_open



locus_ratio = spectral_locus_open / spectral_locus_closed

magenta_ratio = magenta_length / spectral_locus_closed



print("Spectral range: {}nm to {}nm.".format(CMFS.shape.start, CMFS.shape.end))

print("Length of spectra: {} units.".format(spectral_locus_open))

print("Length of alychnae: {} units.".format(magenta_length))

print("Length of total closed spectral locus: {} units.".format(spectral_locus_closed))

print("Ratio of spectra overall: {}%".format(100.0 * locus_ratio))

print("Alychnae line ratio: {}%".format(100.0 * magenta_ratio))
colourspace = colour.RGB_COLOURSPACE["sRGB"]

red_xy = colourspace.primaries[0]

green_xy = colourspace.primaries[1]



colour.plotting.plot_chromaticity_diagram_CIE1931(

)