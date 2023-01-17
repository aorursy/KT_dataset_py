import pandas as pd

tp = pd.read_csv("../input/tile_placements.csv.gz")
tp.head()
tp = tp.sort_values(by='ts').reset_index(drop=True)
tp.head()
import numpy as np



def place_at_timestamp(tp, ts):

    """

    Uses the raw data source to construct a dataset representing the state of r/Place at the

    given UNIX timestamp.

    """

    pivot = np.argmin(tp['ts'] < ts)

    history = tp.iloc[:pivot]

    

    def HTMLColorToRGB(cs):

        """

        Converts #RRGGBB to an (R, G, B) tuple.

        """

        cs = cs.strip()

        if cs[0] == '#': cs = cs[1:]

        if len(cs) != 6:

            raise ValueError("input #%s is not in #RRGGBB format" % cs)

        r, g, b = cs[:2], cs[2:4], cs[4:]

        r, g, b = [int(n, 16) for n in (r, g, b)]

        return (r, g, b)

    

    def HTMLColorToPercentRGB(cs):

        """

        Converts #RRGGBB to a (Red, Green, Blue) ratio-out-of-1 tuple, as used by matplotlib.

        """

        return tuple(c / 256 for c in HTMLColorToRGB(cs))

    

    html_colormap = [

        '#FFFFFF', '#E4E4E4', '#888888', '#222222', '#FFA7D1', '#E50000', '#E59500', '#A06A42',

        '#E5D900', '#94E044', '#02BE01', '#00E5F0', '#0083C7', '#0000EA', '#E04AFF', '#820080'

    ]

    rgb_colormap = {n: HTMLColorToPercentRGB(html_cs) for n, html_cs in enumerate([

        '#FFFFFF', '#E4E4E4', '#888888', '#222222', '#FFA7D1', '#E50000', '#E59500', '#A06A42',

        '#E5D900', '#94E044', '#02BE01', '#00E5F0', '#0083C7', '#0000EA', '#E04AFF', '#820080'

    ])}

    

    curr = (history

                .groupby(['x_coordinate', 'y_coordinate'])

                .last()

                .color

                .reset_index()

                .pipe(lambda df: df.assign(html_color=df.color.map({n: c for n, c in enumerate(html_colormap)})))

                .pipe(lambda df: df.assign(rgb_color = df.color.map(rgb_colormap)))

           )

    

    return curr



def construct_image_matrix(tiles):

    """

    Given the tile data provided by the `place_at_timestamp` function, returns r/Place at that time

    as a matrix.

    """

    mat = np.zeros((1001, 1001, 3))

    for coord, val in zip(tiles.values[:, 0:2], tiles.values[:, 4]):

        mat[coord[1], coord[0]] = val

    return mat
%time pat = place_at_timestamp(tp, 1490918688000 + (1491238734000 - 1490918688000) / 2)
pat.head()
%time pat_mat = construct_image_matrix(pat)
pat_mat.shape
import matplotlib.pyplot as plt



plt.imshow(pat_mat)
fig = plt.figure(figsize=(24, 24))

plt.imshow(

    construct_image_matrix(

        # r/Place at the end of its history.

        place_at_timestamp(tp, 1491238734000)

    )

)