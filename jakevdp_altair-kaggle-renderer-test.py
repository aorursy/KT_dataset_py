!pip install --ignore-installed --no-deps --target=. git+http://github.com/altair-viz/altair
import altair as alt
alt.__path__, alt.__version__
alt.renderers.enable('kaggle')
import numpy as np
import pandas as pd

rand = np.random.RandomState(578493)
data = pd.DataFrame({
    'x': pd.date_range('2012-01-01', freq='D', periods=365),
    'y1': rand.randn(365).cumsum(),
    'y2': rand.randn(365).cumsum(),
    'y3': rand.randn(365).cumsum()
})

data = data.melt('x')
data.head()
chart = alt.Chart(data).mark_line().encode(
    x='x:T',
    y='value:Q',
    color='variable:N'
).interactive(bind_y=False)

chart
chart.encode(color='variable:O')
!rm -rf altair/ altair-2.3.0.dev0.dist-info/
