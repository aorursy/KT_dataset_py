## Install kelper

!pip install keplergl
!jupyter --version
# Install nodejs using conda on the environment (will be needed for installing labextension)   

# Other options - https://anaconda.org/conda-forge/nodejs   

!conda install -y -c conda-forge/label/cf202003 nodejs
!node --version
!python --version
!jupyter labextension install @jupyter-widgets/jupyterlab-manager keplergl-jupyter
# Load an empty map

from keplergl import KeplerGl

map_1 = KeplerGl()

map_1