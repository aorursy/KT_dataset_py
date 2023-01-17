from IPython.core.display import HTML,display

import cv2,numpy as np,cufflinks as cf

from plotly.offline import init_notebook_mode

from plotly.offline import iplot

import plotly.graph_objs as go

from skimage import io,color,measure
fpath='../input/image-examples-for-mixed-styles/'

def randi(nmin,nmax): return np.random.randint(nmin,nmax)

def configure_plotly():

    display(HTML('''

<script src="/static/components/requirejs/require.js"></script>

<script>requirejs.config({

paths:{base:"/static/base",

       plotly:"https://cdn.plot.ly/plotly-1.5.1.min.js?noext"}});

</script>'''))
init_notebook_mode(connected=True)
go1=[]; level=.75

img=cv2.imread(fpath+'pattern02.png')

lo1=go.Layout(autosize=False,width=img.shape[1]//1.2,

              height=img.shape[0]//1.2,

              yaxis=dict(autorange='reversed'),

              margin=go.layout.Margin(l=2,r=2,t=2,b=2),

              plot_bgcolor='rgba(0,0,0,0)')

gray_img=color.colorconv.rgb2grey(img)

contours=measure.find_contours(gray_img,level)

for c in contours:

    go1+=[go.Scatter(x=c[:,1],y=c[:,0],

                     marker=dict(size=.01),

                     showlegend=False)]

iplot(go.Figure(data=go1,layout=lo1))
cf.go_offline(); configure_plotly()

init_notebook_mode(connected=False)
go1=[]; pi=np.pi; n=24; t=np.arange(0,2*pi,.1**3*2*pi/n)

a,b,c,d,m=randi(5,11),randi(12,24),randi(25,81),\

randi(216,256),randi(100,300)

lo1=go.Layout(autosize=False,width=500,height=520,

              margin=go.layout.Margin(l=2,r=2,t=2,b=2),

              plot_bgcolor='rgba(0,0,0,1)')

for i in range(n):

    f1=(a+.9*np.cos(b*t+2*pi*i/n))*(1+.1*np.cos(c*t+2*pi*i/n))

    f2=(1+.01*np.cos(d*t+2*pi*i/n))*(1+np.sin(t+2*pi*i/n))

    x=f1*f2*np.cos(t); y=f1*f2*np.sin(t)

    go1+=[go.Scatter(x=x,y=y,showlegend=False)]

iplot(go.Figure(data=go1,layout=lo1))
html_str='''

<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>

<div id='pc1' style='width:500px; height:500px;'/>

<script>TEST=document.getElementById('pc1');

function get_int(min,max) {return Math.floor(Math.random()*(max-min+1))+min;};

function fx(a,b,c,q,n,t,k) {

    var x1=Math.cos(Math.PI*t/n+k*Math.PI/q)+Math.cos(a*Math.PI*t/n+k*Math.PI/q);

    var x2=Math.cos(b*Math.PI*t/n+k*Math.PI/q)+Math.cos(c*Math.PI*t/n+k*Math.PI/q);

    return x1+x2};

function fy(a,b,c,q,n,t,k) {

    var y1=Math.sin(Math.PI*t/n+k*Math.PI/q)-Math.sin(a*Math.PI*t/n+k*Math.PI/q);

    var y2=Math.sin(b*Math.PI*t/n+k*Math.PI/q)-Math.sin(c*Math.PI*t/n+k*Math.PI/q);

    return y1+y2};

function arx(a,b,c,q,n,k) {return Array(2*n+1).fill(k).map((k,t)=>fx(a,b,c,q,n,t,k));};

function ary(a,b,c,q,n,k) {return Array(2*n+1).fill(k).map((k,t)=>fy(a,b,c,q,n,t,k));};

function col(k,q) {return 'rgb('+(k/(2*q+2)).toString()+',0,'+(1-k/(2*q+2)).toString()+')';};

function plt(k) {return Plotly.plot(TEST,[{x:arx(a,b,c,q,n,k),y:ary(a,b,c,q,n,k),

                                          line:{color:col(k,q),width:.3},name:k.toString()}]);};

var a=get_int(5,9),b=get_int(10,14),c=get_int(15,19),q=get_int(3,8),n=get_int(10,350);

for (var k=1; k<2*q+1; k++) {plt(k);}

</script>'''

html_file=open("plotlychart.html","w")

html_file.write(html_str); html_file.close()

HTML('''<div id='data1'><iframe src="plotlychart.html" 

height="520" width="520"></iframe></div>''')