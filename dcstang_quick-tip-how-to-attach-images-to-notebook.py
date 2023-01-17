from IPython.display import Image

Image("../input/bricklayer-mason-plasterer-worker-cartoon-100266268.jpg")
from IPython.core.display import display, HTML, Javascript

import IPython.display



html_string = """

<g id="my_image_goes_here"></g>

"""



js_string = """

require.config({

    paths: {

        d3: "https://d3js.org/d3.v4.min"

     }

 });



require(["d3"], function(d3) {

  

  d3.select("#my_image_goes_here")

      .append("img")

      .attr("src", "http://www.freedigitalphotos.net/images/previews/bricklayer-mason-plasterer-worker-cartoon-100266268.jpg")

      .attr("width", "250px")

      .style("border", "10px solid black");



});

"""



h = display(HTML(html_string))

j = IPython.display.Javascript(js_string)

IPython.display.display_javascript(j)