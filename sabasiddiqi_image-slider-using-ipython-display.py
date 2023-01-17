from IPython.core.display import display, HTML, Javascript
   
html=   """
        <style>
        .mySlides {display:none;}
        </style>
        <img class="mySlides" src="https://github.com/SabaSiddiqi/Backup/blob/master/trees/Slide1.PNG?raw=true">
        <img class="mySlides" src="https://github.com/SabaSiddiqi/Backup/blob/master/trees/Slide2.PNG?raw=true">
        <img class="mySlides" src="https://github.com/SabaSiddiqi/Backup/blob/master/trees/Slide3.PNG?raw=true">
        <img class="mySlides" src="https://github.com/SabaSiddiqi/Backup/blob/master/trees/Slide4.PNG?raw=true">
        <img class="mySlides" src="https://github.com/SabaSiddiqi/Backup/blob/master/trees/Slide5.PNG?raw=true">
        <img class="mySlides" src="https://github.com/SabaSiddiqi/Backup/blob/master/trees/Slide6.PNG?raw=true">
        <button class="w3-button w3-display-left" onclick="plusDivs(-1)">&#10094; Previous</button>
        <button class="w3-button w3-display-right" onclick="plusDivs(+1)">Next &#10095;</button>
        <script>
                var slideIndex = 1;
                showDivs(slideIndex);

                function plusDivs(n) {
                showDivs(slideIndex += n);
                }

                function showDivs(n) {
                    var i;
                    var x = document.getElementsByClassName("mySlides");
                    if (n > x.length) {slideIndex = 1} 
                    if (n < 1) {slideIndex = x.length} ;
                    for (i = 0; i < x.length; i++) {
                        x[i].style.display = "none"; 
                    }
                    x[slideIndex-1].style.display = "block"; 
                }
        </script>

        """

display(HTML(html))
