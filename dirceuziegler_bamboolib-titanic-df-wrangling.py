# Uncomment and run the line below. This may take a while. Please wait until the kernel is idle before you continue.
#!pip install --upgrade bamboolib>=1.4.1
# RUN ME AFTER YOU HAVE RELOADED THE BROWSER PAGE

# Uncomment and run lines below
import bamboolib as bam
import pandas as pd
df = pd.read_csv(bam.titanic_csv)
# Uncomment and run the line below.
df
# bamboolib live code export
df = df.sort_values(by=['Name'], ascending=[None])
df = df.sort_values(by=['Name'], ascending=[True])
df

