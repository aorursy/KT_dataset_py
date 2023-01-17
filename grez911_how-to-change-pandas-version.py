!pip install pandas==0.24
import pandas



# You will see not the desired version.

pandas.__version__
# Now it works.

!python -c "import pandas; print(pandas.__version__)"
# Also you can save this to a script and then run it.

!echo "import pandas; print(pandas.__version__)" >> script.py
!python script.py