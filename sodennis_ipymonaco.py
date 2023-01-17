!pip install ipymonaco=="0.0.21a"

# if notebook < 5.3
!jupyter nbextension enable --py --sys-prefix ipymonaco
from ipymonaco import *
hello = Monaco(value="SELECT * FROM table;", theme="vs-dark", language="sql", readOnly=False)
hello