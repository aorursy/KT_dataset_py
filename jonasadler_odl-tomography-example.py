!conda install -y -c astra-toolbox/label/dev astra-toolbox

!pip install https://github.com/odlgroup/odl/archive/master.zip
import odl

import astra



spc = odl.uniform_discr([-1, -1], [1, 1],  [512, 512])



rt = odl.tomo.RayTransform(spc, odl.tomo.parallel_beam_geometry(spc))



phantom = odl.phantom.shepp_logan(spc, modified=True)



data = rt(phantom)



phantom.show('phantom');

data.show('data');