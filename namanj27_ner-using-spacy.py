import spacy

from spacy import displacy

from collections import Counter

import en_core_web_sm

nlp = en_core_web_sm.load()
doc = nlp('''On 2018 December 21, MAXI/GSC detected a bright hard X-ray transient source,

MAXI J1631-479, in the Norma region (Kobayashi et al., ATEL#12320). The

X-ray pulsar AX J1631.9-4752/IGR J16320-4751 is within the MAXI error circle

and was suggested to be the source producing the bright transient emission.

However, the Swift/BAT position reported for the bright source (GCN#23550)

is 8.4 arcminutes away from the XMM position of AX J1631.9-4752.  As this

is significantly more than the 3 arcminute uncertainty (90% confidence)

in the Swift/BAT position, we obtained two NuSTAR observations on December

28: a 14 ks observation pointed at the Swift/BAT position and a 12 ks observation

pointed at AX J1631.9-4752. 



In the 14 ks NuSTAR observation, we found a very bright source at R.A.

= 247.806 deg, Decl. = -47.805 deg (J2000) with an uncertainty of 15 arcsec.

This position is consistent with the Swift/BAT position and is not consistent

with the position of AX J1631.9-4752.  In addition, a search of the SIMBAD

database does not show any known source consistent with the NuSTAR position.

Thus, we conclude that MAXI J1631-479 is a new X-ray transient. 



The NuSTAR observations were performed when the source was 34 degrees away

from the sun. As the sun is close to the FOV, the star tracker used for

aspect reconstruction was only available approximately for 2.6 ks, but

we were able to obtain the source position with NuSTAR nominal astrometric

accuracy due to the brightness of the source. 



The source is detected across the 3-79 keV NuSTAR bandpass with a count

rate in excess of 600 c/s (FPMA and FPMB combined).  An absorbed disk-blackbody

plus power-law model provides a reasonably good description of the continuum

with a reduced chi2 of 1.25 for 893 degrees of freedom.  The column density

is (2.9+/-0.2)e22 atoms/cm2 (using wilm abundances), the temperature of

the disk-blackbody is 1.12+/-0.01 keV, and the power-law photon index is

2.39+/-0.02 (90% confidence errors).  The 3-79 keV and 2-10 keV absorbed

fluxes are 1.8e-8 erg/cm2/s and 1.7e-8 erg/cm2/s, respectively.  The residuals

show clear evidence for an iron Kalpha emission line, and adding a broad

gaussian with an equivalent width of 90 eV improves the fit significantly.

This demonstrates that we are seeing a reflection component in the spectrum.



The spectral properties of MAXI J1631-479 indicate that it is very likely

to be an accreting black hole in the soft state.  In the future, further

work on the MAXI and Swift/BAT data may provide more information about

the overall evolution of the current outburst.  The source is still in

outburst, and multi-wavelength observations are encouraged.  Based on the

behavior of other black hole transients, the outburst may last for another

month or more.''')

print([(X.text, X.label_) for X in doc.ents])
print([(X, X.ent_iob_, X.ent_type_) for X in doc])
labels = [x.label_ for x in doc.ents]

Counter(labels)
# sentences = [x for x in doc.sents]

# print(sentences[0])
displacy.render(nlp(str(doc)), jupyter=True, style='ent')
# displacy.render(nlp(str(doc)), style='dep', jupyter = True, options = {'distance': 120})