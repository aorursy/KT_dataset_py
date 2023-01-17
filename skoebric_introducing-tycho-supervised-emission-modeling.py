# --- Download ---

# Currently hosted on github (with data)

!git clone https://github.com/skoeb/tycho.git #make sure internet is turned on in kaggle   

#Looking to add to pip/conda-forge soon.



#Install environment with tycho.yml, or requirements.txt
import pandas as pd

pd.set_option("display.max_columns", 200)

merged = pd.read_pickle('tycho/processed/merged_df.pkl')

merged.set_index(['datetime_utc','plant_id_wri'], inplace=True, drop=True)

merged