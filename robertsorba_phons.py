import numpy as np

import pandas as pd



from bokeh.resources import INLINE

import bokeh.io

bokeh.io.output_notebook(INLINE)

from bokeh.layouts import column, row

from bokeh.models import CustomJS, Slider, CDSView, Button, Label, Div, PrintfTickFormatter

from bokeh.plotting import ColumnDataSource, figure, output_file, show

from bokeh.palettes import RdYlBu11

from bokeh.transform import linear_cmap
div = Div(width=640, height=350, width_policy='max', align='center',

          background='white',

          text="""

          <b><br><br>Click the Start Button to begin.</b>

          """, 

          style={'font-size': '200%', 'color': 'darkslategrey',

                'text-align': 'center'})

freqTracker = Div(width=640, height=50, width_policy='max', align='center',

                 text="Stopped",

                 style={'font-size': '250%', 'color': 'seagreen', 

                        'text-align': 'center'})

datadiv = Div(text="Results:<br><br>", width=640, height=250, width_policy='max', align='start')
phon = 0

tones = [25, 50, 75, 100, 150, 200, 350, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 

         7500, 8000, 8500, 9000, 9500, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000]

phons = [1] * len(tones)

phonsSource = ColumnDataSource(dict(x=tones, y=phons))

playbutton = Button(label="Start!", button_type="success", max_width=200, sizing_mode="scale_width")

backbutton = Button(label="Abort!", button_type="danger", max_width=200, sizing_mode="scale_width")

skipbutton = Button(label="Skip", button_type="primary", max_width=200, sizing_mode="scale_width", disabled=True)

volumeslider = Slider(title="Volume [dB]", start=-70, end=0, step=1, value=-60,

                     disabled=True, max_width=640)

playCallback = CustomJS(args=dict(play=playbutton, tones=tones, vol=volumeslider, datadiv=datadiv,

                                 skip=skipbutton, div=div, fq=freqTracker, src=phonsSource, back=backbutton),

                       code="""

    const db = vol.value;

    const gainpercent = Math.pow(10, db / 20.0);

    if(play.label == "Start!"){

        play.label = "Calibrate!";

        div.text = "<br>A 3000 Hz tone is now playing. Adjust your computer volume until you can just barely hear the tone. Then press Calibrate.";   

        fq.text = "Frequency: 3000 Hz";

        

        

        // create web audio api context

        window.audioCtx = new (window.AudioContext || window.webkitAudioContext)();

        

        window.volume = window.audioCtx.createGain();

        window.volume.connect(window.audioCtx.destination);

        window.volume.gain.setValueAtTime(gainpercent, window.audioCtx.currentTime);



        // create Oscillator node

        window.osc = window.audioCtx.createOscillator();



        window.osc.type = 'sine';

        window.osc.frequency.setValueAtTime(3000, window.audioCtx.currentTime); // value in hertz

        window.osc.connect(window.volume);

        window.osc.start();

        //window.osc.stop(window.audioCtx.currentTime + 10);

        

        

        

        

    } else if(play.label == "Calibrate!"){

        play.label = "Record Volume!";

        div.text = "Now, adjust the Volume Slider until you can just barely hear the current tone, the press Record Volume. DO NOT adjust your computer volume or you will need to restart. If you can't hear the tone at all after increasing the volume to the max, press Skip.";

        fq.text = "Frequency: " + tones[0] + " Hz";

        src.data['y'][13] = -60;

        src.change.emit();

        vol.disabled = false;

        skip.disabled = false;

        window.toneIndex = 0;

        window.osc.frequency.setValueAtTime(tones[0], window.audioCtx.currentTime);

        vol.value = -70;

        vol.change.emit();

        



    } else{

        src.data['y'][window.toneIndex] = db;

        src.change.emit();

        window.toneIndex = window.toneIndex + 1;

        if(tones[window.toneIndex] == 3000){

            window.toneIndex = window.toneIndex + 1;

        }

        if(window.toneIndex >= tones.length){

            window.osc.stop();

            play.label = "Done!";

            div.text = "<br>Examine your results in the graph below and answer the questions.";

            fq.text = "Stopped";

            

            for(var i in src.data['y']){

                datadiv.text = datadiv.text + src.data['y'][i] + "<br>";

            }

            

            play.disabled = true;

            skip.disabled = true;

            back.disabled = true;

            vol.disabled = true;

            

            IPython.notebook.kernel.execute("phon = " + src.data['y']);

        } else {

            window.osc.frequency.setValueAtTime(tones[window.toneIndex], window.audioCtx.currentTime);

            fq.text = "Frequency: " + tones[window.toneIndex] + " Hz";

            vol.value = -70;

            vol.change.emit();

             

        }

        

    }

    

    

                       """)

playbutton.js_on_click(playCallback)



backCallback = CustomJS(args=dict(play=playbutton, tones=tones, vol=volumeslider,

                                 div=div, fq=freqTracker),

                       code="""

    window.osc.stop();

    play.label = "Start!";

    div.text = "<br><br>Click the Start Button to begin.";

    fq.text = "Stopped"

    vol.value = -60;

    vol.trigger('change');

                       """)



skipCallback = CustomJS(args=dict(play=playbutton, tones=tones, vol=volumeslider, datadiv=datadiv,

                                 skip=skipbutton, back=backbutton, div=div, src=phonsSource,

                                 fq=freqTracker),

                       code="""

    window.toneIndex = window.toneIndex + 1;

    if(tones[window.toneIndex] == 3000){

        window.toneIndex = window.toneIndex + 1;

    }

    if(window.toneIndex >= tones.length){

        window.osc.stop();

        play.label = "Done!";

        div.text = "<br>Examine your results in the graph below and answer the questions.";

        fq.text = "Stopped";

        

        for(var i in src.data['y']){

                datadiv.text = datadiv.text + src.data['y'][i] + "<br>";

            }

            

        play.disabled = true;

        skip.disabled = true;

        back.disabled = true;

        vol.disabled = true;

        IPython.notebook.kernel.execute("phon = " + src.data['y']);

    } else {

        window.osc.frequency.setValueAtTime(tones[window.toneIndex], window.audioCtx.currentTime);

        vol.value = -70;

        fq.text = "Frequency: " + tones[window.toneIndex] + " Hz"

        vol.change.emit();

         

    }

                       """)



skipbutton.js_on_click(skipCallback)



backbutton.js_on_click(backCallback)



volumeCallback = CustomJS(args=dict(vol=volumeslider, tones=tones),

                       code="""

    const db = vol.value;

    const gainpercent = Math.pow(10, db / 20.0);

    window.volume.gain.setValueAtTime(gainpercent, window.audioCtx.currentTime);

                       """)



volumeslider.js_on_change('value', volumeCallback)



p = figure(width=640, height=480, max_width=640, sizing_mode='scale_width',

          x_axis_type='log', y_range=[-70, 0])

p.line(x='x', y='y', source=phonsSource)

p.circle(x='x', y='y', source=phonsSource, size=10)



p.xaxis.axis_label = "Frequency [Hz]"

p.yaxis.axis_label = "Sound Level [dB]"

p.axis.axis_label_text_font_size = "12pt"

p.axis.axis_label_text_font_style = 'bold'

p.xaxis[0].formatter = PrintfTickFormatter(format="%i")







layout = column(div, freqTracker,

                row(playbutton, skipbutton, backbutton, align='center'), 

                volumeslider, p, datadiv, sizing_mode='scale_width', max_width=640)

bokeh.io.show(layout)