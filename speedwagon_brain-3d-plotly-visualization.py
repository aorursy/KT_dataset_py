import h5py

from skimage import measure

from plotly.offline import iplot

from plotly import figure_factory as FF
def make_mesh(image, step_size=1):

    '''

    :param step_size: bigger number worse detalization but faster

    '''

    p = image.transpose(2, 1, 0)

    threshold = 0

    verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold, step_size=step_size, allow_degenerate=True)

    return verts, faces



def plotly_3d(verts, faces):

    x, y, z = zip(*verts)

    colormap = ['rgb(230, 207, 227)']



    fig = FF.create_trisurf(

        x=x, y=y, z=z,

        plot_edges=True,

        show_colorbar=False,

        colormap=colormap,

        simplices=faces,

        backgroundcolor='rgb(64, 64, 64)',

        title='3D Brain Plot'

    )

    iplot(fig)

    

def plot_brain(img):

    v, f = make_mesh(img)

    plotly_3d(v, f)
fp = '../input/trends-assessment-prediction/fMRI_test/10030.mat'

img = h5py.File(fp,'r')['SM_feature'][()]
plot_brain(img[0])