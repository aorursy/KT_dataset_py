# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

### USER ADDED LIBRARIES



import seaborn as sns; sns.set()



###

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
arrS1 = np.reshape(np.array([0.000,0.111,0.217,0.317,0.409,0.475,0.535,0.589,0.644,0.699,0.752,0.800,0.849,0.895,0.932,0.957,0.969,0.977,0.982,0.987,0.991,0.112,0.218,0.318,0.410,0.476,0.535,0.590,0.645,0.700,0.753,0.801,0.850,0.896,0.933,0.958,0.970,0.978,0.983,0.988,0.992,0.996,0.219,0.318,0.410,0.476,0.536,0.591,0.646,0.701,0.753,0.802,0.850,0.896,0.933,0.958,0.970,0.979,0.984,0.988,0.992,0.997,1.000,0.318,0.410,0.476,0.536,0.591,0.646,0.701,0.753,0.802,0.850,0.896,0.933,0.958,0.970,0.978,0.984,0.988,0.992,0.997,1.000,0.996,0.408,0.474,0.534,0.589,0.644,0.699,0.751,0.800,0.848,0.894,0.931,0.956,0.968,0.977,0.982,0.986,0.990,0.995,0.998,0.994,0.990,0.469,0.529,0.584,0.639,0.694,0.747,0.795,0.844,0.889,0.926,0.951,0.964,0.972,0.977,0.981,0.986,0.990,0.993,0.989,0.985,0.978,0.524,0.578,0.633,0.688,0.741,0.789,0.838,0.884,0.921,0.946,0.958,0.966,0.971,0.976,0.980,0.984,0.988,0.984,0.980,0.972,0.964,0.572,0.627,0.682,0.735,0.783,0.832,0.877,0.914,0.940,0.952,0.960,0.965,0.969,0.974,0.978,0.981,0.978,0.973,0.966,0.958,0.951,0.620,0.675,0.728,0.776,0.825,0.871,0.908,0.933,0.945,0.953,0.958,0.963,0.967,0.971,0.975,0.971,0.966,0.959,0.951,0.944,0.936,0.668,0.721,0.770,0.818,0.864,0.901,0.926,0.938,0.946,0.952,0.956,0.960,0.964,0.968,0.964,0.960,0.952,0.944,0.937,0.929,0.922,0.714,0.763,0.811,0.857,0.894,0.919,0.931,0.939,0.944,0.949,0.953,0.957,0.961,0.957,0.953,0.945,0.937,0.930,0.922,0.915,0.907,0.755,0.803,0.849,0.886,0.911,0.923,0.931,0.937,0.941,0.945,0.949,0.953,0.949,0.945,0.937,0.929,0.922,0.914,0.907,0.899,0.891,0.794,0.840,0.877,0.902,0.914,0.922,0.928,0.932,0.936,0.940,0.944,0.940,0.936,0.928,0.921,0.913,0.905,0.898,0.890,0.883,0.875,0.830,0.867,0.892,0.904,0.912,0.918,0.922,0.926,0.930,0.934,0.930,0.926,0.918,0.910,0.903,0.895,0.888,0.880,0.873,0.865,0.857,0.855,0.881,0.893,0.901,0.906,0.910,0.915,0.919,0.922,0.919,0.914,0.907,0.899,0.891,0.884,0.876,0.869,0.861,0.854,0.846,0.838,0.864,0.877,0.885,0.890,0.894,0.898,0.903,0.906,0.902,0.898,0.890,0.883,0.875,0.868,0.860,0.852,0.845,0.837,0.830,0.822,0.815,0.859,0.867,0.873,0.877,0.881,0.885,0.889,0.885,0.881,0.873,0.866,0.858,0.850,0.843,0.835,0.828,0.820,0.812,0.805,0.797,0.790,0.849,0.854,0.859,0.863,0.867,0.871,0.867,0.862,0.855,0.847,0.840,0.832,0.825,0.817,0.809,0.802,0.794,0.787,0.779,0.771,0.764,0.836,0.840,0.845,0.849,0.852,0.849,0.844,0.837,0.829,0.822,0.814,0.806,0.799,0.791,0.784,0.776,0.768,0.761,0.753,0.746,0.738,0.822,0.827,0.831,0.834,0.830,0.826,0.818,0.811,0.803,0.796,0.788,0.781,0.773,0.765,0.758,0.750,0.743,0.735,0.727,0.720,0.712,0.808,0.812,0.816,0.812,0.807,0.800,0.792,0.785,0.777,0.770,0.762,0.754,0.747,0.739,0.732,0.724,0.716,0.709,0.701,0.694,0.686]),(21,21))

ax = sns.heatmap(arrS1)
arrS2 = np.reshape(np.array([0.000,0.071,0.118,0.156,0.193,0.231,0.268,0.306,0.343,0.380,0.418,0.455,0.493,0.530,0.568,0.605,0.642,0.680,0.717,0.755,0.792,0.079,0.127,0.164,0.202,0.239,0.277,0.314,0.352,0.389,0.426,0.464,0.501,0.539,0.576,0.614,0.651,0.688,0.726,0.763,0.801,0.838,0.135,0.173,0.210,0.248,0.285,0.323,0.360,0.398,0.435,0.472,0.510,0.547,0.585,0.622,0.660,0.697,0.734,0.772,0.809,0.847,0.884,0.181,0.219,0.256,0.294,0.331,0.369,0.406,0.444,0.481,0.518,0.556,0.593,0.631,0.668,0.706,0.743,0.781,0.818,0.855,0.893,0.930,0.228,0.265,0.302,0.340,0.377,0.415,0.452,0.490,0.527,0.564,0.602,0.639,0.677,0.714,0.752,0.789,0.827,0.864,0.901,0.939,0.970,0.274,0.311,0.348,0.386,0.423,0.461,0.498,0.536,0.573,0.610,0.648,0.685,0.723,0.760,0.798,0.835,0.873,0.910,0.947,0.979,0.981,0.320,0.357,0.394,0.432,0.469,0.507,0.544,0.582,0.619,0.656,0.694,0.731,0.769,0.806,0.844,0.881,0.919,0.956,0.987,0.990,0.957,0.366,0.403,0.440,0.478,0.515,0.553,0.590,0.628,0.665,0.703,0.740,0.777,0.815,0.852,0.890,0.927,0.965,0.996,0.998,0.966,0.933,0.405,0.442,0.480,0.517,0.555,0.592,0.629,0.667,0.704,0.742,0.779,0.817,0.854,0.892,0.929,0.966,0.998,1.000,0.968,0.934,0.901,0.442,0.479,0.517,0.554,0.591,0.629,0.666,0.704,0.741,0.779,0.816,0.854,0.891,0.928,0.966,0.997,0.999,0.967,0.934,0.901,0.868,0.479,0.516,0.553,0.591,0.628,0.666,0.703,0.741,0.778,0.816,0.853,0.890,0.928,0.965,0.997,0.999,0.967,0.933,0.900,0.867,0.834,0.516,0.553,0.590,0.628,0.665,0.703,0.740,0.778,0.815,0.852,0.890,0.927,0.965,0.996,0.998,0.966,0.933,0.900,0.866,0.833,0.800,0.552,0.590,0.627,0.665,0.702,0.740,0.777,0.814,0.852,0.889,0.927,0.964,0.996,0.998,0.966,0.932,0.899,0.866,0.833,0.800,0.766,0.589,0.627,0.664,0.702,0.739,0.776,0.814,0.851,0.889,0.926,0.964,0.995,0.997,0.965,0.932,0.899,0.865,0.832,0.799,0.766,0.733,0.626,0.664,0.701,0.739,0.776,0.813,0.851,0.888,0.926,0.963,0.994,0.997,0.964,0.931,0.898,0.865,0.832,0.798,0.765,0.732,0.699,0.663,0.701,0.738,0.775,0.813,0.850,0.888,0.925,0.963,0.994,0.996,0.964,0.931,0.897,0.864,0.831,0.798,0.765,0.732,0.698,0.643,0.691,0.729,0.766,0.804,0.841,0.878,0.916,0.953,0.985,0.987,0.955,0.921,0.888,0.855,0.822,0.789,0.755,0.722,0.689,0.634,0.567,0.712,0.750,0.787,0.825,0.862,0.900,0.937,0.968,0.971,0.938,0.905,0.872,0.839,0.805,0.772,0.739,0.706,0.673,0.617,0.551,0.485,0.730,0.767,0.805,0.842,0.879,0.917,0.948,0.951,0.918,0.885,0.852,0.819,0.785,0.752,0.719,0.686,0.653,0.597,0.531,0.465,0.398,0.747,0.785,0.822,0.859,0.897,0.928,0.930,0.898,0.865,0.832,0.799,0.765,0.732,0.699,0.666,0.633,0.577,0.511,0.444,0.378,0.312,0.765,0.802,0.839,0.877,0.908,0.910,0.878,0.845,0.812,0.779,0.745,0.712,0.679,0.646,0.613,0.557,0.491,0.424,0.358,0.292,0.225]),(21,21))

ax = sns.heatmap(arrS2)
arrS3 = np.reshape(np.array([0.461,0.518,0.574,0.630,0.687,0.743,0.799,0.856,0.912,0.929,0.933,0.936,0.940,0.943,0.946,0.950,0.953,0.957,0.960,0.964,0.967,0.531,0.587,0.643,0.700,0.756,0.812,0.869,0.925,0.942,0.946,0.949,0.953,0.956,0.959,0.963,0.966,0.970,0.973,0.976,0.980,0.943,0.600,0.656,0.713,0.769,0.825,0.882,0.938,0.955,0.959,0.962,0.965,0.969,0.972,0.976,0.979,0.983,0.986,0.989,0.993,0.956,0.906,0.663,0.720,0.776,0.833,0.889,0.945,0.962,0.966,0.969,0.973,0.976,0.980,0.983,0.986,0.990,0.993,0.997,1.000,0.963,0.913,0.863,0.719,0.775,0.832,0.888,0.944,0.962,0.965,0.968,0.972,0.975,0.979,0.982,0.986,0.989,0.992,0.996,0.999,0.962,0.912,0.862,0.812,0.775,0.831,0.887,0.944,0.961,0.964,0.968,0.971,0.974,0.978,0.981,0.985,0.988,0.992,0.995,0.998,0.961,0.911,0.862,0.812,0.762,0.830,0.886,0.943,0.960,0.963,0.967,0.970,0.974,0.977,0.980,0.984,0.987,0.991,0.994,0.998,0.961,0.911,0.861,0.811,0.761,0.711,0.886,0.942,0.959,0.963,0.966,0.969,0.973,0.976,0.980,0.983,0.986,0.990,0.993,0.997,0.960,0.910,0.860,0.810,0.760,0.710,0.660,0.941,0.958,0.962,0.965,0.969,0.972,0.975,0.979,0.982,0.986,0.989,0.993,0.996,0.959,0.909,0.859,0.809,0.759,0.709,0.659,0.609,0.958,0.961,0.964,0.968,0.971,0.975,0.978,0.981,0.985,0.988,0.992,0.995,0.958,0.908,0.858,0.808,0.758,0.708,0.658,0.608,0.559,0.960,0.964,0.967,0.970,0.974,0.977,0.981,0.984,0.987,0.991,0.994,0.957,0.907,0.857,0.808,0.758,0.708,0.658,0.608,0.558,0.508,0.963,0.966,0.970,0.973,0.976,0.980,0.983,0.987,0.990,0.993,0.957,0.907,0.857,0.807,0.757,0.707,0.657,0.607,0.557,0.507,0.457,0.965,0.969,0.972,0.976,0.979,0.982,0.986,0.989,0.993,0.956,0.906,0.856,0.806,0.756,0.706,0.656,0.606,0.556,0.506,0.456,0.406,0.968,0.971,0.975,0.978,0.982,0.985,0.988,0.992,0.955,0.905,0.855,0.805,0.755,0.705,0.655,0.605,0.555,0.505,0.455,0.405,0.355,0.971,0.974,0.977,0.981,0.984,0.988,0.991,0.954,0.904,0.854,0.804,0.754,0.704,0.654,0.604,0.554,0.504,0.455,0.405,0.355,0.305,0.973,0.977,0.980,0.983,0.987,0.990,0.953,0.903,0.853,0.803,0.753,0.703,0.654,0.604,0.554,0.504,0.454,0.404,0.354,0.304,0.254,0.976,0.979,0.983,0.986,0.989,0.952,0.903,0.853,0.803,0.753,0.703,0.653,0.603,0.553,0.503,0.453,0.403,0.353,0.303,0.253,0.203,0.978,0.982,0.985,0.989,0.952,0.902,0.852,0.802,0.752,0.702,0.652,0.602,0.552,0.502,0.452,0.402,0.352,0.302,0.252,0.202,0.152,0.981,0.984,0.988,0.951,0.901,0.851,0.801,0.751,0.701,0.651,0.601,0.551,0.501,0.451,0.401,0.351,0.301,0.251,0.201,0.152,0.102,0.984,0.987,0.950,0.900,0.850,0.800,0.750,0.700,0.650,0.600,0.550,0.500,0.450,0.400,0.351,0.301,0.251,0.201,0.151,0.101,0.051,0.986,0.949,0.899,0.849,0.799,0.749,0.699,0.649,0.599,0.550,0.500,0.450,0.400,0.350,0.300,0.250,0.200,0.150,0.100,0.050,0.000]),(21,21))

ax = sns.heatmap(arrS3)
arrS4 = np.reshape(np.array([0.830,0.834,0.838,0.842,0.845,0.849,0.853,0.857,0.861,0.864,0.868,0.872,0.876,0.880,0.884,0.888,0.891,0.895,0.899,0.903,0.907,0.848,0.852,0.856,0.860,0.864,0.868,0.871,0.875,0.879,0.883,0.887,0.891,0.894,0.898,0.902,0.906,0.910,0.914,0.918,0.921,0.925,0.867,0.871,0.874,0.878,0.882,0.886,0.890,0.894,0.897,0.901,0.905,0.909,0.913,0.917,0.921,0.924,0.928,0.932,0.936,0.940,0.943,0.885,0.889,0.893,0.897,0.900,0.904,0.908,0.912,0.916,0.920,0.923,0.927,0.931,0.935,0.939,0.943,0.947,0.950,0.954,0.958,0.929,0.903,0.907,0.911,0.915,0.919,0.923,0.927,0.930,0.934,0.938,0.942,0.946,0.950,0.953,0.957,0.961,0.965,0.969,0.973,0.943,0.887,0.922,0.926,0.930,0.933,0.937,0.941,0.945,0.949,0.953,0.957,0.960,0.964,0.968,0.972,0.976,0.980,0.983,0.987,0.958,0.901,0.845,0.938,0.942,0.946,0.950,0.954,0.958,0.962,0.965,0.969,0.973,0.977,0.981,0.984,0.988,0.992,0.996,1.000,0.970,0.914,0.858,0.801,0.941,0.945,0.949,0.953,0.957,0.961,0.964,0.968,0.972,0.976,0.980,0.984,0.987,0.991,0.995,0.999,0.969,0.913,0.857,0.801,0.744,0.944,0.948,0.952,0.956,0.960,0.964,0.967,0.971,0.975,0.979,0.983,0.986,0.990,0.994,0.998,0.968,0.912,0.856,0.800,0.743,0.687,0.947,0.951,0.955,0.959,0.963,0.966,0.970,0.974,0.978,0.982,0.986,0.989,0.993,0.997,0.968,0.911,0.855,0.799,0.742,0.686,0.630,0.950,0.954,0.958,0.962,0.966,0.969,0.973,0.977,0.981,0.985,0.989,0.992,0.996,0.967,0.910,0.854,0.798,0.741,0.685,0.629,0.572,0.953,0.957,0.961,0.965,0.968,0.972,0.976,0.980,0.984,0.988,0.991,0.995,0.966,0.910,0.853,0.797,0.740,0.684,0.628,0.572,0.515,0.956,0.960,0.964,0.967,0.971,0.975,0.979,0.983,0.987,0.991,0.994,0.965,0.908,0.852,0.796,0.740,0.683,0.627,0.571,0.514,0.458,0.959,0.963,0.966,0.970,0.974,0.978,0.982,0.986,0.990,0.993,0.964,0.907,0.851,0.795,0.739,0.682,0.626,0.570,0.513,0.457,0.401,0.962,0.966,0.969,0.973,0.977,0.981,0.985,0.989,0.993,0.963,0.907,0.850,0.794,0.738,0.681,0.625,0.569,0.512,0.456,0.400,0.344,0.965,0.968,0.972,0.976,0.980,0.984,0.988,0.992,0.962,0.906,0.849,0.793,0.737,0.681,0.624,0.568,0.511,0.455,0.399,0.343,0.286,0.968,0.971,0.975,0.979,0.983,0.987,0.991,0.961,0.905,0.848,0.792,0.736,0.679,0.623,0.567,0.511,0.454,0.398,0.342,0.285,0.229,0.971,0.974,0.978,0.982,0.986,0.990,0.960,0.904,0.848,0.791,0.735,0.679,0.622,0.566,0.510,0.453,0.397,0.341,0.284,0.228,0.172,0.973,0.977,0.981,0.985,0.989,0.959,0.903,0.847,0.790,0.734,0.678,0.621,0.565,0.509,0.452,0.396,0.340,0.283,0.227,0.171,0.115,0.976,0.980,0.984,0.988,0.958,0.902,0.846,0.789,0.733,0.677,0.620,0.564,0.508,0.452,0.395,0.339,0.283,0.226,0.170,0.114,0.057,0.979,0.983,0.987,0.957,0.901,0.845,0.788,0.732,0.676,0.620,0.563,0.507,0.450,0.394,0.338,0.282,0.225,0.169,0.113,0.056,0.000]),(21,21))

ax = sns.heatmap(arrS4)


#end